import math
import wandb

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
import torchaudio

from functools import partial
from torch import nn
from rfwave.feature_extractors import FeatureExtractor
from rfwave.heads import FourierHead
from rfwave.helpers import plot_spectrogram_to_numpy, plot_attention_to_numpy
from rfwave.loss import MelSpecReconstructionLoss
from rfwave.models import Backbone
from rfwave.modules import safe_log, safe_log10, pseudo_huber_loss
from rfwave.multi_band_processor import PQMFProcessor, STFTProcessor, MeanVarProcessor
from rfwave.rvm import RelativeVolumeMel
from rfwave.lr_schedule import get_cosine_schedule_with_warmup
from rfwave.input import (InputAdaptor, InputAdaptorProject, CtxCharInputAdaptor, Ctx2CharInputAdaptor,
                          E2ECtxCharInputAdaptor, CharInputAdaptor, score_mask)
from rfwave.attention import sequence_mask
from rfwave.helpers import save_code
from rfwave.instantaneous_frequency import compute_phase_loss, compute_phase_error, compute_instantaneous_frequency
from rfwave.feature_weight import get_feature_weight, get_feature_weight2
from rfwave.dit import DiTRFTTSMultiTaskBackbone, DiTRFE2ETTSMultiTaskBackbone
from rfwave.standalone_alignment import (
    StandaloneAlignment, gaussian_prior, compute_alignment_loss,
    compute_attention_distill_loss, duration_from_attention)
from rfwave.logit_normal import LogitNormal
from rfwave.dataset import get_exp_length, get_exp_scale
from rfwave.e2e_duration import E2EDuration, DurModel
from rfwave.multi_band_processor import DurationProcessor


def sequence_mask_with_ctx(length, ctx_start=None, ctx_length=None, max_length=None):
    non_padding = sequence_mask(length + 1, max_length)
    non_padding = non_padding[:, :-1]  # 1 padding frame got trained for layer norm
    if ctx_length is None or ctx_start is None:
        return non_padding
    else:
        assert torch.all(ctx_start + ctx_length < length)
        if max_length is None:
            max_length = length.max()
        non_ctx = torch.arange(0, max_length, device=length.device).unsqueeze(0)
        non_ctx = torch.logical_or(
            non_ctx < ctx_start.unsqueeze(1), non_ctx >= (ctx_start + ctx_length).unsqueeze(1))
        return torch.logical_and(non_ctx, non_padding)


def masked_mse_loss(pred, target, mask=None):
    loss = F.mse_loss(pred, target, reduction='none')
    if mask is None:
        return loss.mean()
    else:
        loss = (loss * mask.unsqueeze(1)).mean(dim=(1, 2))
        mask_factor = mask.size(1) / mask.sum(1)
        return (loss * mask_factor).mean()


class RectifiedFlow(nn.Module):
    def __init__(self, backbon: Backbone, head: FourierHead,
                 num_steps=10, feature_loss=False, wave=False, num_bands=8,
                 intt=0., intt_mode=None, p_uncond=0., guidance_scale=1.):
        super().__init__()
        self.backbone = backbon
        self.head = head
        self.N = num_steps
        self.feature_loss = feature_loss
        # wave: normal -(stft)-> freq_noise -> NN -> freq_feat -(istft)-> wave -> loss
        # freq: normal -> NN -> freq_feat -> loss
        self.wave = wave
        self.equalizer = wave
        self.stft_norm = not wave
        self.stft_loss = False
        self.phase_loss = False
        self.overlap_loss = True
        self.pred1_consistent_loss = True
        self.num_bands = num_bands
        self.num_bins = self.head.n_fft // 2 // self.num_bands
        self.left_overlap = 8
        self.right_overlap = 8
        self.overlap = self.left_overlap + self.right_overlap
        self.cond_mask_right_overlap = True
        self.prev_cond = False
        self.parallel_uncond = True
        self.time_balance_loss = True
        self.noise_alpha = 0.1
        self.cfg = guidance_scale > 1.
        self.p_uncond = p_uncond
        self.guidance_scale = guidance_scale
        self.output_channels1 = self.backbone.output_channels1
        self.output_channels2 = self.backbone.output_channels2
        self.intt = intt
        self.intt_mode = intt_mode
        self.cfg_iter = False
        out_ch = (self.backbone.output_channels if hasattr(self.backbone, 'output_channels')
                  else self.backbone.output_channels2)
        assert out_ch == self.head.n_fft // self.num_bands + 2 * self.overlap
        assert self.wave ^ self.stft_norm
        assert self.right_overlap >= 1  # at least one to deal with the last dimension of fft feature.
        # t_sampling = 'logit_normal' if isinstance(backbon, DiTRFTTSMultiTaskBackbone) else 'uniform'
        t_sampling = 'uniform'
        self.t_dist = LogitNormal(mu=0., sigma=1.) if t_sampling == 'logit_normal' else None
        if self.stft_norm:
            self.stft_processor = MeanVarProcessor(self.head.n_fft + 2)
        if self.equalizer:
            self.eq_processor = PQMFProcessor(subbands=8, taps=124, cutoff_ratio=0.071)
        if self.feature_loss:
            self.register_buffer(
                "feature_weight", get_feature_weight2(self.head.n_fft, self.head.hop_length))
        self.tandem_processor = MeanVarProcessor(self.output_channels1)

    def get_subband(self, S, i):
        # if i.numel() > 1:
        #     i = i[0]
        S = torch.stack(torch.chunk(S, 2, dim=1), dim=-1)
        if i == -1:
            sS = S.new_zeros((S.shape[0], (self.num_bins + self.overlap) * 2, S.shape[2]))
        else:
            pS = F.pad(S, (0, 0, 0, 0, self.left_overlap, self.right_overlap - 1), mode='constant')
            sS = pS[:, i * self.num_bins: (i + 1) * self.num_bins + self.overlap]
            sS = torch.cat([sS[..., 0], sS[..., 1]], dim=1)
        return sS

    def place_subband(self, sS, i):
        # if i.numel() > 1:
        #     i = i[0]
        S = sS.new_zeros([sS.size(0), self.head.n_fft // 2 + self.overlap, sS.size(2), 2])
        rsS, isS = torch.chunk(sS, 2, dim=1)
        S[:, i * self.num_bins: (i + 1) * self.num_bins + self.overlap, :, 0] = rsS
        S[:, i * self.num_bins: (i + 1) * self.num_bins + self.overlap, :, 1] = isS
        S = S[:, self.left_overlap: S.size(1) - self.right_overlap + 1]
        return torch.cat([S[..., 0], S[..., 1]], dim=1)

    def get_joint_subband(self, S):
        S = torch.stack(torch.chunk(S, 2, dim=1), dim=-1)
        S = F.pad(S, (0, 0, 0, 0, self.left_overlap, self.right_overlap - 1), mode='circular')
        # for version before 2.1
        # S = torch.cat([S[:, S.size(1) - self.left_overlap:], S, S[:, :self.right_overlap - 1]], dim=1)
        assert S.size(1) == self.num_bins * self.num_bands + self.overlap
        S = S.unfold(1, self.num_bins + self.overlap, self.num_bins)  # shape (batch_size, num_bands, seq_len, 2, band_dim)
        S = S.permute(0, 1, 4, 2, 3)  # shape (batch_size, num_bands, band_dim, seq_len, 2)
        S = torch.cat([S[..., 0], S[..., 1]], dim=2)
        S = S.reshape(S.size(0), -1, S.size(-1))
        return S

    def place_joint_subband(self, S):
        def _get_subband(s, i):
            if i == self.num_bands - 1:
                return s[:, self.left_overlap: s.size(1) - self.right_overlap + 1]
            else:
                return s[:, self.left_overlap: s.size(1) - self.right_overlap]
        assert S.size(1) == self.num_bands * (self.num_bins + self.overlap) * 2
        sS_ri = torch.chunk(S, self.num_bands * 2, dim=1)
        sS_r = [_get_subband(s, i) for i, s in enumerate(sS_ri[0::2])]
        sS_i = [_get_subband(s, i) for i, s in enumerate(sS_ri[1::2])]
        return torch.cat([torch.cat(sS_r, dim=1), torch.cat(sS_i, dim=1)], dim=1)

    def mask_cond(self, cond):
        cond = torch.stack(torch.chunk(cond, 2, dim=1), dim=-1)
        cond[:, cond.size(1) - self.right_overlap:] = 0.
        return torch.cat([cond[..., 0], cond[..., 1]], dim=1)

    def get_z0(self, text, bandwidth_id, length):
        bandwidth_id = bandwidth_id[0]
        # for training var pred.
        if self.wave:
            nf = length if self.head.padding == "same" else (length - 1)
            r = torch.randn([text.shape[0], self.head.hop_length * nf], device=text.device)
            rf = self.stft(r)
            z0_2 = self.get_subband(rf, bandwidth_id)
        else:
            r = torch.randn([text.shape[0], self.head.n_fft + 2, length], device=text.device)
            z0_2 = self.get_subband(r, bandwidth_id)
        z0_1 = torch.randn([text.shape[0], self.output_channels1, length], device=text.device)
        return torch.cat([z0_1, z0_2], dim=1)

    def same_noise_for_bands(self, noise):
        ns = noise.shape
        noise = noise.reshape(ns[0] // self.num_bands, self.num_bands, *ns[1:])
        noise = noise[:, 0]
        return noise.repeat_interleave(self.num_bands, dim=0)

    def get_joint_z0(self, text, length):
        if self.wave:
            nf = length if self.head.padding == "same" else (length - 1)
            r = torch.randn([text.shape[0], self.head.hop_length * nf], device=text.device)
            rf = self.stft(r)
            z0_2 = self.get_joint_subband(rf)
        else:
            r = torch.randn([text.shape[0], self.head.n_fft + 2, length], device=text.device)
            z0_2 = self.get_joint_subband(r)
        z0_2 = z0_2.reshape(z0_2.size(0) * self.num_bands, z0_2.size(1) // self.num_bands, z0_2.size(2))
        z0_1 = torch.randn((text.size(0) * self.num_bands, self.output_channels1, length),
                           device=text.device)
        z0_1 = self.same_noise_for_bands(z0_1)
        return torch.cat([z0_1, z0_2], dim=1)

    def get_eq_norm_stft(self, audio):
        if self.equalizer:
            audio = self.eq_processor.project_sample(audio.unsqueeze(1)).squeeze(1)
        S = self.stft(audio)
        if self.stft_norm:
            S = self.stft_processor.project_sample(S)
        return S

    def get_z1(self, audio, mel, bandwidth_id):
        bandwidth_id = bandwidth_id[0]
        S = self.get_eq_norm_stft(audio)
        z1_2 = self.get_subband(S, bandwidth_id)
        if self.prev_cond:
            cond_band = self.get_subband(S, bandwidth_id - 1)
            if self.cond_mask_right_overlap:
                cond_band = self.mask_cond(cond_band)
        else:
            cond_band = None
        mel = self.tandem_processor.project_sample(mel)
        z1 = torch.cat([mel, z1_2], dim=1)
        return z1, cond_band

    def get_joint_z1(self, audio, mel):
        S = self.get_eq_norm_stft(audio)
        z1_2 = self.get_joint_subband(S)
        z1_2 = z1_2.reshape(z1_2.size(0) * self.num_bands, z1_2.size(1) // self.num_bands, z1_2.size(2))
        mel = self.tandem_processor.project_sample(mel)
        mel = torch.repeat_interleave(mel, self.num_bands, 0)
        z1 = torch.cat([mel, z1_2], dim=1)
        return z1

    def get_wave(self, x):
        if self.stft_norm:
            x = self.stft_processor.return_sample(x)
        x = self.istft(x)
        if self.equalizer:
            x = self.eq_processor.return_sample(x.unsqueeze(1)).squeeze(1)
        return x

    def get_tandem(self, mel):
        return self.tandem_processor.return_sample(mel)

    def get_intt_zt(self, t_, z0, z1):
        if self.intt == 0.:
            zt = t_ * z1 + (1. - t_) * z0
            return zt

        z0_1, z0_2 = self.split(z0)
        z1_1, z1_2 = self.split(z1)

        if self.intt_mode == 'async':
            m = t_ < self.intt
            intt1 = t_ / self.intt
            zt_1 = intt1 * z1_1 + (1 - intt1) * z0_1
            zt_1 = torch.where(m, zt_1, z1_1)
            zt_2 = t_ * z1_2 + (1. - t_) * z0_2
            zt = torch.cat([zt_1, zt_2], dim=1)
        elif self.intt_mode == 'cascade':
            m = t_ < self.intt
            intt1 = t_ / self.intt
            intt2 = (t_ - self.intt) / (1 - self.intt)
            # z_t = t_ * z1 + (1. - t_) * z0
            zt_1 = intt1 * z1_1 + (1 - intt1) * z0_1
            zt_2 = intt2 * z1_2 + (1 - intt2) * z0_2
            zt_1 = torch.where(m, zt_1, z1_1)
            zt_2 = torch.where(m, z0_2, zt_2)
            zt = torch.cat([zt_1, zt_2], dim=1)
        elif self.intt_mode == 'pipeline':
            m1 = t_ < 1. - self.intt
            m2 = t_ < self.intt
            intt1 = t_ / (1 - self.intt)
            intt2 = (t_ - self.intt) / (1 - self.intt)
            zt_1 = intt1 * z1_1 + (1 - intt1) * z0_1
            zt_2 = intt2 * z1_2 + (1 - intt2) * z0_2
            zt_1 = torch.where(m1, zt_1, z1_1)
            zt_2 = torch.where(m2, z0_2, zt_2)
            zt = torch.cat([zt_1, zt_2], dim=1)
        else:
            raise ValueError(f'Unsupported intt mode {self.intt_mode}')
        return zt

    def get_intt_target(self, t_, z0, z1):
        if self.intt == 0.:
            target = z1 - z0
            return target

        z0_1, z0_2 = self.split(z0)
        z1_1, z1_2 = self.split(z1)
        target_1 = z1_1 - z0_1
        target_2 = z1_2 - z0_2
        zero_1 = torch.zeros_like(target_1)
        zero_2 = torch.zeros_like(target_2)

        if self.intt_mode == 'async':
            m = t_ < self.intt
            target_1 = torch.where(m, target_1, zero_1)
            target = torch.cat([target_1, target_2], dim=1)
        elif self.intt_mode == 'cascade':
            m = t_ < self.intt
            # target = z1 - z0
            target_1 = torch.where(m, target_1, zero_1)
            target_2 = torch.where(m, zero_2, target_2)
            target = torch.cat([target_1, target_2], dim=1)
        elif self.intt_mode == 'pipeline':
            m1 = t_ < 1. - self.intt
            m2 = t_ < self.intt
            target_1 = torch.where(m1, target_1, zero_1)
            target_2 = torch.where(m2, zero_2, target_2)
            target = torch.cat([target_1, target_2], dim=1)
        else:
            raise ValueError(f'Unsupported intt mode {self.intt_mode}')
        return target

    def sample_t(self, shape, device):
        if self.t_dist is not None:
            if self.intt == 0:
                return self.t_dist.sample(shape).to(device)
            elif self.intt_mode == 'async':
                return self.t_dist.sample(shape).to(device)
            elif self.intt_mode == 'cascade':
                return torch.where(
                    torch.rand(shape) < self.intt, self.t_dist.sample(shape) * self.intt,
                    self.intt + self.t_dist.sample(shape) * (1 - self.intt)).to(device)
            elif self.intt_mode == 'pipeline':
                return self.t_dist.sample(shape).to(device)
            else:
                raise ValueError(f'Unsupported intt mode {self.intt_mode}')
        else:
            return torch.rand(shape, device=device)

    def get_train_tuple(self, text, mel, audio_input):
        if self.prev_cond or not self.parallel_uncond:
            t = self.sample_t((mel.size(0),), device=mel.device)
            bandwidth_id = torch.tile(torch.randint(0, self.num_bands, (), device=mel.device), (mel.size(0),))
            bandwidth_id = torch.ones([mel.shape[0]], dtype=torch.long, device=mel.device) * bandwidth_id
            z0 = self.get_z0(text, bandwidth_id, mel.size(2))
            z1, cond_band = self.get_z1(audio_input, mel, bandwidth_id)
            text = torch.cat([text, cond_band], 1) if self.prev_cond else mel
        else:
            t = self.sample_t((mel.size(0),), device=mel.device).repeat_interleave(self.num_bands, 0)
            z0 = self.get_joint_z0(text, mel.size(2))
            z1 = self.get_joint_z1(audio_input, mel)
            bandwidth_id = torch.tile(torch.arange(self.num_bands, device=mel.device), (mel.size(0),))
            text = torch.repeat_interleave(text, self.num_bands, 0)
        t_ = t.view(-1, 1, 1)
        zt = self.get_intt_zt(t_, z0, z1)
        target = self.get_intt_target(t_, z0, z1)
        return text, bandwidth_id, (zt, t, target)

    def get_pred(self, z_t, t, text, bandwidth_id, **kwargs):
        _backbone_keys = ['start', 'length', 'num_tokens', 'ctx_start', 'ctx_length',
                          'token_exp_scale', 'standalone_attn', 'token', 'duration']
        backbone_kwargs = {}
        for k in _backbone_keys:
            if k in kwargs and kwargs[k] is not None:
                n_rpt = z_t.size(0) // kwargs[k].size(0)
                assert n_rpt == self.num_bands
                backbone_kwargs[k] = torch.repeat_interleave(kwargs[k], n_rpt, 0)
        pred = self.backbone(z_t, t, text, bandwidth_id, **backbone_kwargs)
        return pred if isinstance(pred, tuple) else (pred, None, None)

    def make_pred1_consistent(self, pred1):
        pred1 = pred1.reshape(pred1.size(0) // self.num_bands, self.num_bands, *pred1.shape[1:])
        # pred1 = pred1.mean(dim=1, keepdim=True)  # same mel for different bands, average
        pred1 = pred1[:, 0: 1]  # same mel for different bands, the first band
        pred1 = torch.repeat_interleave(pred1, self.num_bands, dim=1).reshape(-1, *pred1.shape[2:])
        return pred1

    def get_t_dt(self, N):
        if self.intt == 0:
            ts = np.linspace(0., 1., N + 1)
            dt = ts[1:] - ts[:-1]
        elif self.intt_mode == 'async':
            ts = np.linspace(0., 1., N + 1)
            dt = ts[1:] - ts[:-1]
        elif self.intt_mode == 'cascade':
            ts1 = np.linspace(0., 1, int(N * self.intt), endpoint=False)
            ts2 = np.linspace(0, 1., int(N * (1 - self.intt)) + 1)
            ts12 = np.concatenate([ts1, 1 + ts2])
            ts = np.linspace(0., 1., N + 1)
            dt = ts12[1:] - ts12[:-1]
        elif self.intt_mode == 'pipeline':
            ts = np.linspace(0, 1., N + 1)
            dt = (ts[1:] - ts[:-1]) / (1 - self.intt)
        else:
            raise ValueError(f'Unsupported intt mode {self.intt_mode}')
        return np.stack([ts[:-1], dt], axis=1)

    def intt_postprocess(self, pred1, pred2, t):
        if self.intt > 0.:
            if self.intt_mode == 'async':
                if t < self.intt:
                    pred1 = pred1 / self.intt
                else:
                    pred1 = torch.zeros_like(pred1)
            elif self.intt_mode == 'cascade':
                if t < self.intt:
                    pred2 = torch.zeros_like(pred2)
                else:
                    pred1 = torch.zeros_like(pred1)
            elif self.intt_mode == 'pipeline':
                if t < self.intt:
                    pred2 = torch.zeros_like(pred2)
                elif t > 1. - self.intt:
                    pred1 = torch.zeros_like(pred1)
            else:
                raise ValueError(f'Unsupported intt mode {self.intt_mode}')
        return pred1, pred2

    @torch.no_grad()
    def sample_ode_subband(self, text, band, bandwidth_id, N=None, keep_traj=False, **kwargs):
        if self.intt > 0.:
            assert (self.intt / (1. / N)) % 1 == 0
        ### NOTE: Use Euler method to sample from the learned flow
        if N is None:
            N = self.N
        traj_1 = []  # to store the trajectory
        traj_2 = []  # to store the trajectory
        attns = []
        if self.prev_cond or not self.parallel_uncond:
            assert band is not None
            assert bandwidth_id is not None
            # get z0 must be called before pre-processing text
            z0 = self.get_z0(text, bandwidth_id, kwargs['out_length'])
            text = torch.cat([text, band], 1) if self.prev_cond else text
        else:
            assert band is None
            assert bandwidth_id is None
            bandwidth_id = torch.tile(torch.arange(self.num_bands, device=text.device), (text.size(0),))
            z0 = self.get_joint_z0(text, kwargs['out_length'])  # get z0 must be called before pre-processing text
            text = torch.repeat_interleave(text, self.num_bands, 0)

        if self.cfg:
            text = torch.cat([text, torch.ones_like(text) * text.mean(dim=(0, 2), keepdim=True)], dim=0)
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    kwargs[k] = torch.cat([v] * 2, dim=0)

        z = z0.detach()
        fs = (z.size(0) // self.num_bands, self.output_channels2 * self.num_bands, z.size(2))
        ss = (z.size(0), self.output_channels2, z.size(2))
        t_dt = self.get_t_dt(N)
        for i, (t, dt) in enumerate(t_dt):
            t_ = torch.ones(z.size(0)) * t
            if self.cfg:
                (z_, t_, bandwidth_id_) = [torch.cat([v] * 2, dim=0) for v in (z, t_, bandwidth_id)]
                pred, opt_attn, _ = self.get_pred(z_, t_.to(text.device), text, bandwidth_id_, **kwargs)
                pred, uncond_pred = torch.chunk(pred, 2, dim=0)
                pred = uncond_pred + self.guidance_scale * (pred - uncond_pred)
            else:
                pred, opt_attn, _ = self.get_pred(z, t_.to(text.device), text, bandwidth_id, **kwargs)
            if self.wave:
                pred1, pred2 = self.split(pred)
                if self.prev_cond or not self.parallel_uncond:
                    pred2 = self.place_subband(pred2, bandwidth_id[0])
                    pred2 = self.stft(self.istft(pred2))
                    pred2 = self.get_subband(pred2, bandwidth_id[0])
                else:
                    pred2 = self.place_joint_subband(pred2.reshape(fs))
                    pred2 = self.stft(self.istft(pred2))
                    pred2 = self.get_joint_subband(pred2).reshape(ss)
            else:
                pred1, pred2 = self.split(pred)
            pred1 = self.make_pred1_consistent(pred1)
            pred1, pred2 = self.intt_postprocess(pred1, pred2, t)
            pred = torch.cat([pred1, pred2], dim=1)
            z = z.detach() + pred * dt
            if i == N - 1 or keep_traj:
                z_1, z_2 = self.split(z.detach())
                traj_1.append(z_1)
                traj_2.append(z_2)
                attns.append(opt_attn)
        return traj_1, traj_2, attns

    def combine_subbands(self, traj):
        assert len(traj) == self.num_bands

        def _reshape(x):
            return torch.stack(torch.chunk(x, 2, dim=1), dim=-1)

        def _combine(l):
            c_x = []
            for i, x_i in enumerate(l):
                if i == len(l) - 1:
                    x_i = _reshape(x_i)[:, self.left_overlap: x_i.size(1) // 2 - self.right_overlap + 1]
                else:
                    x_i = _reshape(x_i)[:, self.left_overlap: x_i.size(1) // 2 - self.right_overlap]
                c_x.append(x_i)
            c_x = torch.cat(c_x, dim=1)
            return torch.cat([c_x[..., 0], c_x[..., 1]], dim=1)

        c_traj = []
        for traj_bands in zip(*traj):
            c_traj.append(_combine(traj_bands))

        return c_traj

    def sample_ode(self, text, N=None, keep_traj=False, **kwargs):
        assert 'out_length' in kwargs, "out_length (#output frames) must be kwargs in sample ode"
        traj_1 = []
        traj_2 = []
        attns = []
        if self.prev_cond or not self.parallel_uncond:
            band = text.new_zeros(
                (text.shape[0], 2 * (self.num_bins + self.overlap), text.shape[2]), device=text.device)
            for i in range(self.num_bands):
                bandwidth_id = torch.ones([text.shape[0]], dtype=torch.long, device=text.device) * i
                traj_i_1, traj_i_2, attn_i = self.sample_ode_subband(
                    text, band, bandwidth_id, N=N, keep_traj=keep_traj, **kwargs)
                band = traj_i_1[-1]
                if self.prev_cond:
                    band = torch.zeros_like(band)
                elif self.cond_mask_right_overlap:
                    band = self.mask_cond(band)
                traj_2.append(traj_i_2)
                traj_1 = traj_i_1  # traj_i_2 is the same for different bands.
                attns = attn_i
            traj_2 = self.combine_subbands(traj_2)
        else:
            traj_1, traj_2, attns = self.sample_ode_subband(
                text, None, None, N=N, keep_traj=keep_traj, **kwargs)
            rbs = traj_1[0].size(0) // self.num_bands
            traj_1 = [tt.reshape(rbs, self.num_bands, *tt.shape[1:])[:, 0] for tt in traj_1]
            traj_2 = [self.place_joint_subband(tt.reshape(rbs, -1, tt.size(2)))
                      for tt in traj_2]
        return [self.get_tandem(tt) for tt in traj_1], [self.get_wave(tt) for tt in traj_2], attns

    def stft(self, wave):
        S = self.head.get_spec(wave.float()) / np.sqrt(self.head.n_fft).astype(np.float32)
        return torch.cat([S.real, S.imag], dim=1).type_as(wave)

    def istft(self, S):
        S = S * np.sqrt(self.head.n_fft).astype(np.float32)
        r, i = torch.chunk(S.float(), 2, dim=1)
        c = r + 1j * i
        return self.head.get_wave(c).type_as(S)

    def time_balance_for_loss(self, pred, target):
        v = target.var(dim=1, keepdim=True)
        pred = pred / torch.sqrt(v + 1e-6)
        target = target / torch.sqrt(v + 1e-6)
        return pred, target

    def _place_diff(self, diff, bandwidth_id):
        if self.prev_cond or not self.parallel_uncond:
            diff = self.place_subband(diff, bandwidth_id[0])
        else:
            diff = diff.reshape(diff.shape[0] // self.num_bands, -1, diff.shape[2])
            diff = self.place_joint_subband(diff)
        return diff

    def compute_stft_loss(self, z_t, t, target, pred, bandwidth_id):
        def _mag(S):
            r, i = torch.chunk(S, 2, dim=1)
            return torch.sqrt(r ** 2 + i ** 2)

        z0 = z_t - t.view(-1, 1, 1) * target
        pred_z1 = self._place_diff(z0 + pred, bandwidth_id)
        target_z1 = self._place_diff(z0 + target, bandwidth_id)
        pred_mag = _mag(pred_z1)
        target_mag = _mag(target_z1)
        pred_log_mag = safe_log10(pred_mag)
        target_log_mag = safe_log10(target_mag)
        mag_loss = F.mse_loss(pred_log_mag, target_log_mag)
        converge_loss = (torch.norm(pred_mag - target_mag, p="fro") /
                         (torch.norm(target_mag, p="fro") + 1))
        return mag_loss + converge_loss

    def compute_phase_loss(self, z_t, t, target, pred, bandwidth_id):
        def _complex_spec(S):
            r, i = torch.chunk(S, 2, dim=1)
            return torch.complex(r, i)

        z0 = z_t - t.view(-1, 1, 1) * target
        pred_z1 = self._place_diff(z0 + pred, bandwidth_id)
        target_z1 = self._place_diff(z0 + target, bandwidth_id)
        pred_spec = _complex_spec(pred_z1)
        targ_spec = _complex_spec(target_z1)
        pred_if = compute_instantaneous_frequency(pred_spec)
        targ_if = compute_instantaneous_frequency(targ_spec)
        phase_loss = compute_phase_loss(pred_if, targ_if)
        return phase_loss

    def compute_rf_loss1(self, pred, target, mask_factor=None):
        loss = F.mse_loss(pred, target, reduction='none').mean(dim=(1, 2))
        return loss.mean() if mask_factor is None else (mask_factor * loss).mean()

    def compute_rf_loss2(self, pred, target, bandwidth_id, mask_factor=None):
        if self.time_balance_loss:
            pred, target = self.time_balance_for_loss(pred, target)
        if self.wave:
            diff = pred - target
            diff = self._place_diff(diff, bandwidth_id)
            loss = self.istft(diff).pow(2.).mean(dim=(1,))
        else:
            if self.feature_loss:
                diff = (pred - target) * np.sqrt(self.head.n_fft).astype(np.float32)
                diff = self._place_diff(diff, bandwidth_id)
                diff = torch.einsum("bct,dc->bdt", diff, self.feature_weight)
                feature_loss = torch.mean(diff ** 2, dim=(1, 2)) * self.num_bands
                loss = feature_loss
            else:
                loss = F.mse_loss(pred, target, reduction='none').mean(dim=(1, 2))
        if mask_factor is not None:
            rpt = mask_factor.size(0) // loss.size(0)
            mask_factor = mask_factor.reshape(loss.size(0), rpt)[:, 0]
            return (mask_factor * loss).mean()
        else:
            return loss.mean()

    def split(self, feat):
        return torch.split(feat, [self.output_channels1, self.output_channels2], dim=1)

    def compute_overlap_loss(self, pred):
        if self.prev_cond or not self.parallel_uncond:
            return 0.

        def _overlap_loss(pred):
            loss = 0.
            d = pred[0].size(1)
            for i in range(self.num_bands):
                if i == 0 or self.left_overlap == 0:
                    left_loss = 0.
                else:
                    cur_left = pred[i][:, :self.left_overlap]
                    prev_no_pad = pred[i - 1][:, self.left_overlap: d - self.right_overlap]
                    prev_right = prev_no_pad[:, prev_no_pad.size(1) - self.left_overlap:]
                    left_loss = F.mse_loss(cur_left, prev_right)
                if i == self.num_bands - 1 or self.right_overlap == 0:
                    right_loss = 0.
                else:
                    cur_right = pred[i][:, d - self.right_overlap:]
                    next_no_pad = pred[i + 1][:, self.left_overlap: d - self.right_overlap]
                    next_left = next_no_pad[:, :self.right_overlap]
                    right_loss = F.mse_loss(cur_right, next_left)
                loss = loss + left_loss + right_loss
            return loss

        pred = pred.reshape(pred.shape[0] // self.num_bands, -1, pred.shape[2])
        pred_ri = torch.chunk(pred, self.num_bands * 2, dim=1)
        pred_r, pred_i = pred_ri[::2], pred_ri[1::2]
        loss_i = _overlap_loss(pred_i)
        loss_r = _overlap_loss(pred_r)
        return (loss_i + loss_r) / self.num_bands

    def compute_pred1_consistent_loss(self, pred1):
        pred1 = pred1.reshape(pred1.size(0) // self.num_bands, self.num_bands, *pred1.shape[1:])
        pred1_roll = torch.roll(pred1, 1, dims=1)
        loss = F.mse_loss(pred1, pred1_roll) * self.num_bands
        return loss

    def get_intt_pred(self, t_, pred):
        pred1, pred2 = self.split(pred)
        if self.intt == 0.:
            return pred1, pred2
        zero1 = torch.zeros_like(pred1)
        zero2 = torch.zeros_like(pred2)
        if self.intt_mode == 'async':
            m = t_ < self.intt
            pred1 = torch.where(m, pred1, zero1)
        elif self.intt_mode == 'cascade':
            m = t_ < self.intt
            pred1 = torch.where(m, pred1, zero1)
            pred2 = torch.where(m, zero2, pred2)
        elif self.intt_mode == 'pipeline':
            m1 = t_ < 1. - self.intt
            m2 = t_ < self.intt
            pred1 = torch.where(m1, pred1, zero1)
            pred2 = torch.where(m2, zero2, pred2)
        else:
            raise ValueError(f'Unsupported intt mode {self.intt_mode}')
        return pred1, pred2

    def compute_alignment_loss(self, opt_attn, **kwargs):
        if self.backbone.rad_align:
            assert 'num_tokens' in kwargs and 'token_exp_scale' in kwargs and opt_attn is not None
            assert 'global_step' in kwargs
            attn_loss = compute_alignment_loss(
                opt_attn, kwargs['num_tokens'], kwargs['token_exp_scale'],
                ref_length=60 if kwargs['global_step'] < 50000 else None)
        elif self.backbone.standalone_align and self.backbone.standalone_distill:
            assert 'standalone_attn' in kwargs and kwargs['standalone_attn'] is not None
            attn_loss = compute_attention_distill_loss(opt_attn, kwargs['standalone_attn'])
        else:
            attn_loss = 0.
        return attn_loss

    def compute_loss(self, z_t, t, target, text, bandwidth_id, mask, **kwargs):
        if self.cfg and np.random.uniform() < self.p_uncond:
            # not grad back to the alignment module.
            text = torch.ones_like(text) * text.detach().mean(dim=(0, 2), keepdim=True)
            self.cfg_iter = True
        else:
            self.cfg_iter = False

        pred, opt_attn, ctx = self.get_pred(z_t, t, text, bandwidth_id, **kwargs)
        if mask is not None:
            pred = torch.where(mask.unsqueeze(1), pred, target)
            mask_factor = mask.size(1) / mask.sum(1)
        else:
            mask_factor = None

        pred, z_t, t, target = [v.float() for v in (pred, z_t, t, target)]
        opt_attn = opt_attn.float() if opt_attn is not None else None
        t_ = t.view(-1, 1, 1)
        z_t1, z_t2 = self.split(z_t)
        pred1, pred2 = self.get_intt_pred(t_, pred)
        target1, target2 = self.split(target)
        stft_loss = self.compute_stft_loss(z_t2, t, target2, pred2, bandwidth_id) if self.stft_loss else 0.
        phase_loss = self.compute_phase_loss(z_t2, t, target2, pred2, bandwidth_id) if self.phase_loss else 0.
        loss1 = self.compute_rf_loss1(pred1, target1, mask_factor)
        loss2 = self.compute_rf_loss2(pred2, target2, bandwidth_id, mask_factor)
        overlap_loss = self.compute_overlap_loss(pred2) if self.overlap_loss else 0.
        pred1_consistent_loss = self.compute_pred1_consistent_loss(pred1) if self.pred1_consistent_loss else 0.
        attn_loss = self.compute_alignment_loss(opt_attn, **kwargs) * (0. if self.cfg_iter else 1.)
        loss_dict = {"loss1": loss1, "loss2": loss2, "stft_loss": stft_loss, "phase_loss": phase_loss,
                     "overlap_loss": overlap_loss, "pred1_consistent_loss": pred1_consistent_loss,
                     "attn_loss": attn_loss, "attn": opt_attn, 'ctx': ctx}
        return (loss1 * 5. + loss2 +
                (stft_loss + phase_loss + overlap_loss + attn_loss + pred1_consistent_loss) * 0.1, loss_dict)


class VocosExp(pl.LightningModule):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        backbone: Backbone,
        head: FourierHead,
        input_adaptor: InputAdaptor = None,
        standalone_align: StandaloneAlignment = None,
        sample_rate: int = 24000,
        initial_learning_rate: float = 2e-4,
        feature_loss: bool = False,
        wave: bool = False,
        num_bands: int = 8,
        intt: float = 0.,
        intt_mode: str = None,
        guidance_scale: float = 1.,
        p_uncond: float = 0.2,
        num_warmup_steps: int = 0,
        torch_compile: bool = False,
        aux_loss: bool = False,
    ):
        """
        Args:
            feature_extractor (FeatureExtractor): An instance of FeatureExtractor to extract features from audio signals.
            backbone (Backbone): An instance of Backbone model.
            head (FourierHead):  An instance of Fourier head to generate spectral coefficients and reconstruct a waveform.
            sample_rate (int): Sampling rate of the audio signals.
            initial_learning_rate (float): Initial learning rate for the optimizer.
            num_warmup_steps (int): Number of steps for the warmup phase of learning rate scheduler. Default is 0.
            mel_loss_coeff (float, optional): Coefficient for Mel-spectrogram loss in the loss function. Default is 45.
            mrd_loss_coeff (float, optional): Coefficient for Multi Resolution Discriminator loss. Default is 1.0.
            pretrain_mel_steps (int, optional): Number of steps to pre-train the model without the GAN objective. Default is 0.
            decay_mel_coeff (bool, optional): If True, the Mel-spectrogram loss coefficient is decayed during training. Default is False.
            evaluate_utmos (bool, optional): If True, UTMOS scores are computed for each validation run.
            evaluate_pesq (bool, optional): If True, PESQ scores are computed for each validation run.
            evaluate_periodicty (bool, optional): If True, periodicity scores are computed for each validation run.
        """
        super().__init__()
        assert intt == 0. or (0. < intt < 1. and intt_mode in ('cascade', 'async') or
                              0. < intt < 0.5 and intt_mode == 'pipeline')
        print(f"using intt {intt:.2f}, intt_mode {intt_mode}")
        self.save_hyperparameters(ignore=["feature_extractor", "backbone", "head", "input_adaptor",
                                          "input_adaptor_proj", "standalone_align"])
        self.feature_extractor = feature_extractor

        if torch_compile:
            # input_adaptor = torch.compile(input_adaptor)
            # backbone = torch.compile(backbone)
            backbone.module = torch.compile(backbone.module)

        self.input_adaptor = input_adaptor
        self.reflow = RectifiedFlow(
            backbone, head, feature_loss=feature_loss, wave=wave, num_bands=num_bands,
            intt=intt, intt_mode=intt_mode, guidance_scale=guidance_scale, p_uncond=p_uncond)
        self.standalone_align = standalone_align

        self.train_dur = False
        self.standalone_dur = None
        if standalone_align is not None or backbone.rad_align or backbone.e2_tts:
            assert isinstance(getattr(backbone, "_orig_mod", backbone), DiTRFE2ETTSMultiTaskBackbone)
            assert not (backbone.rad_align and (backbone.standalone_align or standalone_align is not None)), (
                "standalone align and rad align should not be used at the same time.")

            self.dur_output_exp_scale = backbone.standalone_distill or backbone.rad_align or backbone.e2_tts
            self.standalone_dur = E2EDuration(
                DurModel(self.input_adaptor.dim, 2),
                output_exp_scale=self.dur_output_exp_scale)
            self.dur_processor = DurationProcessor() if self.dur_output_exp_scale else MeanVarProcessor(1)
            self.standalone_dur_start_step = 10000
            self.train_dur = True

        assert input_adaptor is not None
        self.tandem_type = 'mel'
        self.aux_loss = aux_loss  # aux_loss improve attention learning for E2E
        assert not (self.aux_loss and backbone.e2_tts)  # not use aux_loss for e2-tts
        # aux_input_dim = (backbone.dim if isinstance(backbone, DiTRFE2ETTSMultiTaskBackbone)
        #                  else self.input_adaptor.dim)
        aux_input_dim = self.input_adaptor.dim
        self.input_adaptor_proj = (InputAdaptorProject(aux_input_dim, feature_extractor.dim)
                                   if self.aux_loss else None)

        self.melspec_loss = MelSpecReconstructionLoss(sample_rate=sample_rate)
        self.rvm = RelativeVolumeMel(sample_rate=sample_rate)

        self.validation_step_outputs = []
        self.automatic_optimization = False
        assert num_bands == backbone.num_bands
        self.num_skipped_nan = 0

    def configure_optimizers(self):
        gen_params = [
            {"params": self.feature_extractor.parameters()},
            {"params": self.reflow.backbone.parameters()},
            {"params": self.reflow.head.parameters()},
            {"params": self.input_adaptor.parameters()},
        ]
        if self.input_adaptor_proj is not None:
            gen_params.append({"params": self.input_adaptor_proj.parameters()})
        if self.standalone_align is not None:
            gen_params.append({"params": self.standalone_align.parameters()})
        if self.standalone_dur is not None:
            gen_params.append({"params": self.standalone_dur.parameters()})
        opt_gen = torch.optim.AdamW(gen_params, lr=self.hparams.initial_learning_rate)

        max_steps = self.trainer.max_steps  # // 2  # Max steps per optimizer
        scheduler_gen = get_cosine_schedule_with_warmup(
            opt_gen, num_warmup_steps=self.hparams.num_warmup_steps, num_training_steps=max_steps,
        )
        return [opt_gen], [{"scheduler": scheduler_gen, "interval": "step"}]

    def skip_nan(self, optimizer):
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = torch.isfinite(param.grad).all()
                if not valid_gradients:
                    break
        if not valid_gradients:
            self.num_skipped_nan += 1
            # print("detected inf or nan values in gradients. not updating model parameters")
            optimizer.zero_grad()

    def get_log_spec(self, audio):
        S = self.reflow.head.get_spec(audio)
        target = torch.log(S.abs().clamp_min_(1e-6))
        return target

    def compute_aux_loss(self, features, audio_input, mask):
        rpt = features.size(0) // audio_input.size(0)
        target = self.feature_extractor(audio_input)
        target = target.repeat_interleave(rpt, 0)
        pred = self.input_adaptor_proj(features)
        if pred.ndim == 4:
            target = target.unsqueeze(1)  # for num heads
        loss = masked_mse_loss(pred, target, mask)
        return loss

    def process_context(self, phone_info):
        pi_kwargs = {}
        ctx_kwargs = {}
        input_adaptor = getattr(self.input_adaptor, "_orig_mod", self.input_adaptor)
        if isinstance(input_adaptor, CharInputAdaptor):
            assert len(phone_info) == 4
            pi_kwargs['start'] = phone_info[3]
            length = phone_info[1].sum(1)
            pi_kwargs['length'] = length
            ctx_kwargs['length'] = length
        elif isinstance(input_adaptor, E2ECtxCharInputAdaptor):
            assert len(phone_info) == 6
            phone_info[2] = self.feature_extractor(phone_info[2])
            # num_tokens * epx_scale to get num_frames
            # length = torch.round(phone_info[1] * phone_info[5]).long()
            length = get_exp_length(phone_info[1], phone_info[5])
            pi_kwargs['num_tokens'] = phone_info[1]
            pi_kwargs['ctx_start'] = phone_info[3]
            pi_kwargs['ctx_length'] = phone_info[4]
            pi_kwargs['token_exp_scale'] = phone_info[5]
            ctx_kwargs['length'] = length
            ctx_kwargs['ctx_start'] = phone_info[3]
            ctx_kwargs['ctx_length'] = phone_info[4]
            phone_info = [phone_info[0], phone_info[2]]
        elif isinstance(input_adaptor, CtxCharInputAdaptor):
            assert len(phone_info) == 10
            phone_info[5] = self.feature_extractor(phone_info[5])
            pi_kwargs['start'] = phone_info[3]
            pi_kwargs['length'] = phone_info[4]
            ctx_kwargs['length'] = phone_info[4]
            ctx_kwargs['ctx_start'] = phone_info[6]
            ctx_kwargs['ctx_length'] = phone_info[7]
            # delete elements so phone info can be used as arguments of input adaptor
            del phone_info[6]
            del phone_info[4]
        elif isinstance(input_adaptor, Ctx2CharInputAdaptor):
            assert len(phone_info) == 10
            phone_info[5] = self.feature_extractor(phone_info[5])
            pi_kwargs['start'] = phone_info[3]
            pi_kwargs['length'] = phone_info[4]
            ctx_kwargs['length'] = phone_info[4]
            ctx_kwargs['ctx_start'] = phone_info[6]
            ctx_kwargs['ctx_length'] = phone_info[7]
            # delete elements so phone info can be used as arguments of input adaptor
            del phone_info[6]
            del phone_info[4]
        else:
            raise ValueError(f"Invalid phone_info, #fields {len(phone_info)}")
        return phone_info, pi_kwargs, ctx_kwargs

    def compute_sa_align(self, mel, tokens, num_tokens, token_exp_scale, ctx_length, *args, **kwargs):
        # tokens contains text token and reference mel.
        tokens, _ = torch.split(tokens, [num_tokens.max(), ctx_length.max()], dim=2)
        mask = score_mask(num_tokens)
        attn_prior = gaussian_prior(num_tokens, token_exp_scale)
        attn = self.standalone_align(mel, tokens, token_exp_scale, mask, attn_prior=attn_prior)
        sa_loss = compute_alignment_loss(attn, num_tokens, token_exp_scale)
        return attn, sa_loss

    def compute_dur_loss(self, text, standalone_attn, **kwargs):
        num_tokens = kwargs['num_tokens']
        ref_length = kwargs['ctx_length']
        dur_out = self.standalone_dur(text, num_tokens, ref_length)
        token_exp_scale = kwargs['token_exp_scale']
        if self.dur_output_exp_scale:
            token_exp_scale = self.dur_processor.project_sample(token_exp_scale)
            loss = F.mse_loss(dur_out, token_exp_scale)
        else:
            length = get_exp_length(num_tokens, token_exp_scale)
            dur = duration_from_attention(standalone_attn, num_tokens, length)
            mask = sequence_mask(num_tokens)
            dur_out = torch.where(mask, dur_out, dur)
            dur = self.dur_processor.project_sample(dur)
            loss = F.mse_loss(dur_out, dur, reduction='none').mean(-1) * num_tokens.max() / num_tokens
            loss = loss.mean()
        if self.global_step < self.standalone_dur_start_step:
            return loss * 0.  # all weights are used.
        else:
            return loss

    def on_before_optimizer_step(self, optimizer):
        # Note: `unscale` happens after the closure is executed, but before the `on_before_optimizer_step` hook.
        self.skip_nan(optimizer)
        self.clip_gradients(optimizer, gradient_clip_val=1., gradient_clip_algorithm="norm")

    @property
    def cfg_iter(self):
        return self.reflow.cfg_iter

    def training_step(self, batch, batch_idx, **kwargs):
        audio_input, phone_info = batch
        phone_info, pi_kwargs, ctx_kwargs = self.process_context(phone_info)
        mel = self.feature_extractor(audio_input, **kwargs)
        tandem_feat = mel if self.tandem_type == 'mel' else self.get_log_spec(audio_input)
        # train generator
        opt_gen = self.optimizers()
        sch_gen = self.lr_schedulers()

        opt_gen.zero_grad()
        text = self.input_adaptor(*phone_info)
        sa_attn, sa_loss = (self.compute_sa_align(mel, text, **pi_kwargs)
                            if self.standalone_align else (None, 0.))
        dur_loss = self.compute_dur_loss(text, standalone_attn=sa_attn, **pi_kwargs) if self.train_dur else 0.
        text_ext, bandwidth_id, (z_t, t, target) = self.reflow.get_train_tuple(text, tandem_feat, audio_input)
        kwargs.update(**pi_kwargs)
        kwargs['global_step'] = self.global_step
        ctx_mask = sequence_mask_with_ctx(**ctx_kwargs)
        ctx_mask = ctx_mask.repeat_interleave(z_t.size(0) // ctx_mask.size(0), 0)
        loss, loss_dict = self.reflow.compute_loss(
            z_t, t, target, text_ext, bandwidth_id=bandwidth_id, mask=ctx_mask,
            standalone_attn=sa_attn, **kwargs)
        aux = loss_dict['ctx'] if loss_dict['ctx'] is not None else text
        # TODO: attn_loss = 0. is cfg iter, but this is not correct when gt duration is used.
        # not mask reference ctx for attn aux loss mask since aux feature is calculated from
        # attention weights @ text embedding
        mask = sequence_mask_with_ctx(ctx_kwargs['length'])
        mask = mask.repeat_interleave(z_t.size(0) // mask.size(0), 0)
        cond_mel_loss = self.compute_aux_loss(aux, audio_input, mask) if self.aux_loss else 0.
        cfg_or_not = 0. if self.cfg_iter else 1.
        loss = loss + (sa_loss + dur_loss + cond_mel_loss) * 0.1 * cfg_or_not
        self.manual_backward(loss)
        opt_gen.step()
        sch_gen.step()

        self.log("train/total_loss", loss, prog_bar=True, logger=False)
        if self.global_step % 1000 == 0 and self.global_rank == 0:
            with torch.no_grad():
                kwargs['out_length'] = z_t.size(2)
                kwargs['standalone_attn'] = sa_attn
                mel_hat_traj, audio_hat_traj, _ = self.reflow.sample_ode(text, N=100, **kwargs)
            audio_hat = audio_hat_traj[-1]
            mel_hat = mel_hat_traj[-1]
            mask = sequence_mask(ctx_kwargs['length']) if 'length' in ctx_kwargs else None
            tandem_mel_loss = masked_mse_loss(mel_hat, tandem_feat, mask)
            mel_loss = masked_mse_loss(self.feature_extractor(audio_hat), mel, mask)
            self.logger.log_metrics(
                {"train/mel_loss": mel_loss, "train/tandem_mel_loss": tandem_mel_loss,
                 "train/total_loss": loss, "train/cond_mel_loss": cond_mel_loss}, step=self.global_step)
            attn = loss_dict.pop('attn', None)
            loss_dict['sa_attn_loss'] = sa_loss
            loss_dict['dur_loss'] = dur_loss
            if attn is not None:
                attn = attn[0, 0].detach().cpu().numpy()
                title = f'step = {self.global_step},  t = {t[0].item():.5f}, cfg = {self.cfg_iter}'
                self.logger.experiment.log(
                    {"train_media/attn": wandb.Image(plot_attention_to_numpy(attn, title=title))},
                    step=self.global_step)
            loss_dict = dict((f'train/{k}', v) for k, v in loss_dict.items() if 'ctx' not in k)
            self.logger.log_metrics(loss_dict, step=self.global_step)
            rvm_loss = self.rvm(audio_hat.unsqueeze(1), audio_input.unsqueeze(1))
            for k, v in rvm_loss.items():
                self.logger.log_metrics({f"train/{k}": v}, step=self.global_step)
        return loss

    def infer_dur(self, text, **kwargs):
        if self.standalone_dur is None:
            return {}
        num_tokens = kwargs['num_tokens']
        ref_length = kwargs['ctx_length']
        dur_out = self.standalone_dur(text, num_tokens, ref_length)
        dur_out = self.dur_processor.return_sample(dur_out)
        if self.dur_output_exp_scale:
            length = get_exp_length(num_tokens, dur_out)
            return {'out_length': length.clamp(0).max(), 'token_exp_scale': dur_out.clamp(0)}
        else:
            mask = sequence_mask(num_tokens)
            dur_out = dur_out * mask
            dur_out = (torch.ceil(dur_out.clamp_min(0.)).long() if self.global_step < 100
                       else torch.round(dur_out.clamp_min(0.)).long())
            length = dur_out.sum(-1)
            return {'out_length': length.clamp(0).max(), 'duration': dur_out.clamp(0),
                    'token_exp_scale': get_exp_scale(num_tokens, length)}

    def attn_or_dur(self, mel, text, **pi_kwargs):
        if self.standalone_dur is None:
            return {}
        elif mel is not None and self.global_step < self.standalone_dur_start_step * 1.5:
            sa_attn, sa_loss = (self.compute_sa_align(mel, text, **pi_kwargs)
                                if self.standalone_align else (None, 0.))
            return {'standalone_attn': sa_attn}
        else:
            return self.infer_dur(text, **pi_kwargs)

    def validation_step(self, batch, batch_idx, **kwargs):
        audio_input, phone_info = batch
        phone_info, pi_kwargs, ctx_kwargs = self.process_context(phone_info)
        with torch.no_grad():
            mel = self.feature_extractor(audio_input, **kwargs)
            tandem_feat = mel if self.tandem_type == "mel" else self.get_log_spec(audio_input)
            text = self.input_adaptor(*phone_info)
            kwargs.update(**pi_kwargs)
            kwargs['out_length'] = mel.size(2)
            # out_length, token_exp_scale will be replaced.
            aod_kwargs = self.attn_or_dur(mel, text, **pi_kwargs)
            kwargs.update(**aod_kwargs)
            mel_hat_traj, audio_hat_traj, _ = self.reflow.sample_ode(text, N=100, **kwargs)
        audio_hat = audio_hat_traj[-1]
        mel_hat = mel_hat_traj[-1]
        # NOTE: interpolate to calculate loss. not correct. MCD is too time-consuming
        audio_hat_interp = F.interpolate(audio_hat.unsqueeze(1), size=audio_input.shape[1:]).squeeze(1)
        mel_hat_interp = F.interpolate(mel_hat, size=mel.shape[2:])

        mask = sequence_mask(ctx_kwargs['length']) if 'length' in ctx_kwargs else None
        tandem_mel_loss = masked_mse_loss(mel_hat_interp, tandem_feat, mask)
        mel_loss = masked_mse_loss(self.feature_extractor(audio_hat_interp), mel, mask)
        rvm_loss = self.rvm(audio_hat_interp.unsqueeze(1), audio_input.unsqueeze(1))
        phase_loss = compute_phase_error(audio_hat_interp, audio_input, self.reflow.head.get_spec)

        output = {
            "audio_input": audio_input[0],
            "audio_pred": audio_hat[0],
            "mel_pred": mel_hat[0],
            "tandem_mel_loss": tandem_mel_loss,
            "mel_loss": mel_loss,
            "phase_loss": phase_loss,
        }
        output.update(rvm_loss)
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        tandem_mel_loss = torch.stack([x["tandem_mel_loss"] for x in outputs]).mean()
        mel_loss = torch.stack([x["mel_loss"] for x in outputs]).mean()
        phase_loss = torch.stack([x["phase_loss"] for x in outputs]).mean()
        rvm_loss_dict = {}
        for k in outputs[0].keys():
            if k.startswith("rvm"):
                rvm_loss_dict[f'valid/{k}'] = torch.stack([x[k] for x in outputs]).mean()

        self.log("val_loss", mel_loss, sync_dist=True, logger=False)
        if self.global_rank == 0:
            audio_in, audio_pred, tandem_mel_hat = (
                outputs[0]['audio_input'].float(), outputs[0]['audio_pred'].float(), outputs[0]['mel_pred'].float())
            mel_target = self.feature_extractor(audio_in)
            mel_hat = self.feature_extractor(audio_pred)
            metrics = {
                "valid/tandem_mel_loss": tandem_mel_loss,
                "valid/mel_loss": mel_loss,
                "valid/phase_loss": phase_loss}
            self.logger.log_metrics({**metrics, **rvm_loss_dict}, step=self.global_step)
            self.logger.experiment.log(
                {"valid_media/audio_in": wandb.Audio(audio_in.data.cpu().numpy(), sample_rate=self.hparams.sample_rate),
                 "valid_media/audio_hat": wandb.Audio(audio_pred.data.cpu().numpy(), sample_rate=self.hparams.sample_rate),
                 "valid_media/mel_in": wandb.Image(plot_spectrogram_to_numpy(mel_target.data.cpu().numpy())),
                 "valid_media/tandem_mel_hat": wandb.Image(plot_spectrogram_to_numpy(tandem_mel_hat.data.cpu().numpy())),
                 "valid_media/mel_hat": wandb.Image(plot_spectrogram_to_numpy(mel_hat.data.cpu().numpy()))},
                step=self.global_step)
        self.validation_step_outputs.clear()

    # def on_train_epoch_start(self, *args):
    #     torch.cuda.empty_cache()
    def on_train_epoch_end(self) -> None:
        if self.num_skipped_nan > 0:
            print(f"Epoch {self.current_epoch} skipped {self.num_skipped_nan} NAN steps")
        self.num_skipped_nan = 0

    def on_train_start(self, *args):
        if self.global_rank == 0:
            code_fp = save_code(None, self.logger.save_dir)
            # backup code to wandb
            artifact = wandb.Artifact(code_fp.stem, type='code')
            artifact.add_file(str(code_fp), name=code_fp.name)
            self.logger.experiment.log_artifact(artifact)
