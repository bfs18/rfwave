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
from rfwave.helpers import plot_spectrogram_to_numpy, save_figure_to_numpy
from rfwave.loss import MelSpecReconstructionLoss
from rfwave.models import Backbone
from rfwave.modules import safe_log, safe_log10, pseudo_huber_loss
from rfwave.multi_band_processor import MultiBandProcessor, PQMFProcessor, STFTProcessor
from rfwave.rvm import RelativeVolumeMel
from rfwave.lr_schedule import get_cosine_schedule_with_warmup
from rfwave.input import InputAdaptor, InputAdaptorProject
from rfwave.helpers import save_code
from rfwave.instantaneous_frequency import compute_phase_loss, compute_phase_error, compute_instantaneous_frequency
from rfwave.feature_weight import get_feature_weight, get_feature_weight2


class RectifiedFlow(nn.Module):
    def __init__(self, backbon: Backbone, head: FourierHead,
                 num_steps=10, feature_loss=False, wave=False, num_bands=8, intt=0., p_uncond=0., guidance_scale=1.):
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
        self.stft_loss = wave
        self.phase_loss = wave
        self.overlap_loss = True
        self.num_bands = num_bands
        self.num_bins = self.head.n_fft // 2 // self.num_bands
        self.left_overlap = 0
        self.right_overlap = 1
        self.overlap = self.left_overlap + self.right_overlap
        self.cond_mask_right_overlap = True
        self.time_balance_loss = True
        self.noise_alpha = 0.1
        self.cfg = guidance_scale > 1.
        self.p_uncond = p_uncond
        self.guidance_scale = guidance_scale
        self.output_channels1 = self.backbone.output_channels1
        self.output_channels2 = self.backbone.output_channels2
        self.intt = intt  # resemble cascade system
        assert self.wave ^ self.stft_norm
        assert self.right_overlap >= 1  # at least one to deal with the last dimension of fft feature.
        if self.stft_norm:
            self.stft_processor = STFTProcessor(self.head.n_fft + 2)
        if self.equalizer:
            self.eq_processor = PQMFProcessor(subbands=8, taps=124, cutoff_ratio=0.071)
        if self.feature_loss:
            self.register_buffer(
                "feature_weight", get_feature_weight2(self.head.n_fft, self.head.hop_length))
        self.tandem_processor = STFTProcessor(self.output_channels1)

    def get_subband(self, S, i):
        if i.numel() > 1:
            i = i[0]
        S = torch.stack(torch.chunk(S, 2, dim=1), dim=-1)
        if i == -1:
            sS = S.new_zeros((S.shape[0], (self.num_bins + self.overlap) * 2, S.shape[2]))
        else:
            pS = F.pad(S, (0, 0, 0, 0, self.left_overlap, self.right_overlap - 1), mode='constant')
            sS = pS[:, i * self.num_bins: (i + 1) * self.num_bins + self.overlap]
            sS = torch.cat([sS[..., 0], sS[..., 1]], dim=1)
        return sS

    def place_subband(self, sS, i):
        if i.numel() > 1:
            i = i[0]
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

    def same_noise_for_bands(self, noise):
        ns = noise.shape
        noise = noise.reshape(ns[0] // self.num_bands, self.num_bands, *ns[1:])
        noise = noise[:, 0]
        return noise.repeat_interleave(self.num_bands, dim=0)

    def get_joint_z0(self, text, l):
        if self.wave:
            nf = l if self.head.padding == "same" else (l - 1)
            r = torch.randn([text.shape[0], self.head.hop_length * nf], device=text.device)
            rf = self.stft(r)
            z0_2 = self.get_joint_subband(rf)
        else:
            r = torch.randn([text.shape[0], self.head.n_fft + 2, l], device=text.device)
            z0_2 = self.get_joint_subband(r)
        z0_2 = z0_2.reshape(z0_2.size(0) * self.num_bands, z0_2.size(1) // self.num_bands, l)
        z0_1 = torch.randn((text.size(0) * self.num_bands, self.output_channels1, l),
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
        z0_1, z0_2 = self.split(z0)
        z1_1, z1_2 = self.split(z1)
        m = t_ < self.intt
        intt1 = t_ / self.intt
        intt2 = (t_ - self.intt) / (1 - self.intt)
        # z_t = t_ * z1 + (1. - t_) * z0
        zt_1 = intt1 * z1_1 + (1 - intt1) * z0_1
        zt_2 = intt2 * z1_2 + (1 - intt2) * z0_2
        zt_1 = torch.where(m, zt_1, z1_1)
        zt_2 = torch.where(m, z0_2, zt_2)
        zt = torch.cat([zt_1, zt_2], dim=1)
        return zt

    def get_intt_target(self, t_, z0, z1):
        z0_1, z0_2 = self.split(z0)
        z1_1, z1_2 = self.split(z1)
        m = t_ < self.intt
        # target = z1 - z0
        target_1 = z1_1 - z0_1
        target_2 = z1_2 - z0_2
        zero_1 = torch.zeros_like(target_1)
        zero_2 = torch.zeros_like(target_2)
        target_1 = torch.where(m, target_1, zero_1)
        target_2 = torch.where(m, zero_2, target_2)
        target = torch.cat([target_1, target_2], dim=1)
        return target

    def sample_t(self, shape, device):
        return torch.rand(shape, device=device)

    def repeat_cond(self, cond):
        return [torch.repeat_interleave(c, self.num_bands, 0) for c in cond]

    def cfg_cond(self, cond, train=True):
        if train:
            cond[0] = torch.ones_like(cond[0]) * cond[0].mean(dim=(0, 2), keepdim=True)
            cond[2] = torch.ones_like(cond[2]) * cond[2].mean(dim=(0, 2), keepdim=True)
        else:
            cond[0] = torch.cat(
                [cond[0], torch.ones_like(cond[0]) * cond[0].mean(dim=(0, 2), keepdim=True)], dim=0)
            cond[3] = torch.cat(
                [cond[3], torch.ones_like(cond[3]) * cond[3].mean(dim=(0, 2), keepdim=True)], dim=0)
            (cond[1], cond[2], cond[4]) = [
                torch.cat([v] * 2, dim=0) for v in (cond[1], cond[2], cond[4])]
        return cond

    def get_train_tuple(self, cond, mel, audio_input):
        assert len(cond) == 6
        cond, length = cond[:-1], cond[-1]
        t = self.sample_t((mel.size(0),), device=mel.device).repeat_interleave(self.num_bands, 0)
        z0 = self.get_joint_z0(cond[0], length)
        z1 = self.get_joint_z1(audio_input, mel)
        bandwidth_id = torch.tile(torch.arange(self.num_bands, device=mel.device), (mel.size(0),))
        cond = self.repeat_cond(cond)
        t_ = t.view(-1, 1, 1)
        if self.intt > 0.:
            zt = self.get_intt_zt(t_, z0, z1)
            target = self.get_intt_target(t_, z0, z1)
        else:
            zt = t_ * z1 + (1. - t_) * z0
            target = z1 - z0
        return cond, bandwidth_id, (zt, t, target)

    def get_pred(self, z_t, t, cond, bandwidth_id):
        pred = self.backbone.tts_forward(z_t, t, cond, bandwidth_id)
        return pred

    def get_intt_dt(self, t, dt):
        if t < self.intt:
            dt = dt / self.intt
        else:
            dt = dt / (1 - self.intt)
        return dt

    def get_ts(self, N):
        ts = torch.linspace(0., 1., N + 1)
        return ts

    @torch.no_grad()
    def sample_ode_subband(self, cond, band, bandwidth_id, N=None, keep_traj=False):
        assert len(cond) == 6
        if self.intt > 0.:
            assert (self.intt / (1. / N)) % 1 == 0
        ### NOTE: Use Euler method to sample from the learned flow
        if N is None:
            N = self.N
        traj_1 = []  # to store the trajectory
        traj_2 = []  # to store the trajectory

        assert band is None
        assert bandwidth_id is None
        bandwidth_id = torch.tile(torch.arange(self.num_bands, device=cond[0].device), (cond[0].size(0),))
        cond, length = cond[:-1], cond[-1]
        z0 = self.get_joint_z0(cond[0], length)  # get z0 must be called before pre-processing text
        cond = self.repeat_cond(cond)

        ts = self.get_ts(N)
        z = z0.detach()
        fs = (z.size(0) // self.num_bands, self.output_channels2 * self.num_bands, z.size(2))
        ss = (z.size(0), self.output_channels2, z.size(2))
        for i, t in enumerate(ts[:-1]):
            dt = ts[i + 1] - t
            dt = self.get_intt_dt(t, dt) if self.intt > 0. else dt
            t_ = torch.ones(z.size(0)) * t
            if self.cfg:
                cond = self.repeat_cond(cond, False)
                (z_, t_, bandwidth_id_) = [torch.cat([v] * 2, dim=0) for v in (z, t_, bandwidth_id)]
                pred = self.get_pred(z_, t_.to(cond[0].device), cond, bandwidth_id_)
                pred, uncond_pred = torch.chunk(pred, 2, dim=0)
                pred = uncond_pred + self.guidance_scale * (pred - uncond_pred)
            else:
                pred = self.get_pred(z, t_.to(cond[0].device), cond, bandwidth_id)
            if self.wave:
                pred1, pred2 = self.split(pred)
                pred2 = self.place_joint_subband(pred2.reshape(fs))
                pred2 = self.stft(self.istft(pred2))
                pred2 = self.get_joint_subband(pred2).reshape(ss)
            else:
                pred1, pred2 = self.split(pred)
            if self.intt > 0.:
                if t < self.intt:
                    pred2 = torch.zeros_like(pred2)
                else:
                    pred1 = torch.zeros_like(pred1)
            pred = torch.cat([pred1, pred2], dim=1)
            z = z.detach() + pred * dt
            if i == N - 1 or keep_traj:
                z_1, z_2 = self.split(z.detach())
                traj_1.append(z_1)
                traj_2.append(z_2)
        return traj_1, traj_2

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

    def sample_ode(self, cond, N=None, keep_traj=False):
        traj_1, traj_2 = self.sample_ode_subband(
            cond, None, None, N=N, keep_traj=keep_traj)
        rbs = traj_1[0].size(0) // self.num_bands
        traj_1 = [tt.reshape(rbs, self.num_bands, *tt.shape[1:])[:, 0] for tt in traj_1]
        traj_2 = [self.place_joint_subband(tt.reshape(rbs, -1, tt.size(2)))
                  for tt in traj_2]
        return [self.get_tandem(tt) for tt in traj_1], [self.get_wave(tt) for tt in traj_2]

    def stft(self, wave):
        S = self.head.get_spec(wave) / np.sqrt(self.head.n_fft).astype(np.float32)
        return torch.cat([S.real, S.imag], dim=1)

    def istft(self, S):
       S = S * np.sqrt(self.head.n_fft).astype(np.float32)
       r, i = torch.chunk(S, 2, dim=1)
       c = r + 1j * i
       return self.head.get_wave(c)

    def time_balance_for_loss(self, pred, target):
        v = target.var(dim=1, keepdim=True)
        pred = pred / torch.sqrt(v + 1e-6)
        target = target / torch.sqrt(v + 1e-6)
        return pred, target

    def _place_diff(self, diff, bandwidth_id):
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
        converge_loss = torch.mean(torch.norm(pred_mag - target_mag, p="fro", dim=[1, 2]) /
                                   torch.norm(target_mag, p="fro", dim=[1, 2]))
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

    def compute_rf_loss1(self, pred, target):
        return F.mse_loss(pred, target)

    def compute_rf_loss2(self, pred, target, bandwidth_id):
        if self.wave:
            if self.time_balance_loss:
                pred, target = self.time_balance_for_loss(pred, target)
            diff = pred - target
            diff = self._place_diff(diff, bandwidth_id)
            loss = self.istft(diff).pow(2.).mean()
        else:
            if self.feature_loss:
                if self.time_balance_loss:
                    pred, target = self.time_balance_for_loss(pred, target)

                diff = (pred - target) * np.sqrt(self.head.n_fft).astype(np.float32)
                diff = self._place_diff(diff, bandwidth_id)
                diff = torch.einsum("bct,dc->bdt", diff, self.feature_weight)
                feature_loss = torch.mean(diff ** 2) * self.num_bands
                loss = feature_loss
            else:
                if self.time_balance_loss:
                    pred, target = self.time_balance_for_loss(pred, target)
                loss = F.mse_loss(pred, target)
        return loss

    def split(self, feat):
        return torch.split(feat, [self.output_channels1, self.output_channels2], dim=1)

    def compute_overlap_loss(self, pred):
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

    def get_intt_pred(self, t_, pred):
        pred1, pred2 = self.split(pred)
        m = t_ < self.intt
        zero1 = torch.zeros_like(pred1)
        zero2 = torch.zeros_like(pred2)
        pred1 = torch.where(m, pred1, zero1)
        pred2 = torch.where(m, zero2, pred2)
        return pred1, pred2

    def compute_loss(self, z_t, t, target, cond, bandwidth_id):
        if self.cfg and np.random.uniform() < self.p_uncond:
            cond = self.cfg_cond(cond, True)
        pred = self.get_pred(z_t, t, cond, bandwidth_id)
        t_ = t.view(-1, 1, 1)
        z_t1, z_t2 = self.split(z_t)
        if self.intt > 0.:
            pred1, pred2 = self.get_intt_pred(t_, pred)
        else:
            pred1, pred2 = self.split(pred)
        target1, target2 = self.split(target)
        stft_loss = self.compute_stft_loss(z_t2, t, target2, pred2, bandwidth_id) if self.stft_loss else 0.
        phase_loss = self.compute_phase_loss(z_t2, t, target2, pred2, bandwidth_id) if self.phase_loss else 0.
        loss1 = self.compute_rf_loss1(pred1, target1)
        loss2 = self.compute_rf_loss2(pred2, target2, bandwidth_id)
        overlap_loss = self.compute_overlap_loss(pred2) if self.overlap_loss else 0.
        loss_dict = {"loss1": loss1, "loss2": loss2, "stft_loss": stft_loss,
                     "phase_loss": phase_loss, "overlap_loss": overlap_loss}
        return loss1 * 5. + loss2 + stft_loss + phase_loss + overlap_loss * 0.1, loss_dict


class VocosExp(pl.LightningModule):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        backbone: Backbone,
        head: FourierHead,
        input_adaptor: InputAdaptor = None,
        sample_rate: int = 24000,
        initial_learning_rate: float = 2e-4,
        feature_loss: bool = False,
        wave: bool = False,
        num_bands: int = 8,
        intt: float = 0.,
        guidance_scale: float = 1.,
        p_uncond: float = 0.2,
        num_warmup_steps: int = 0,
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
        assert 0. <= intt < 1.
        if intt > 0.:
            print(f"using intt {intt:.2f}")
        self.save_hyperparameters(ignore=["feature_extractor", "backbone", "head", "input_adaptor"])
        self.feature_extractor = feature_extractor
        self.input_adaptor = input_adaptor
        self.reflow = RectifiedFlow(
            backbone, head, feature_loss=feature_loss, wave=wave, num_bands=num_bands, intt=intt,
            guidance_scale=guidance_scale, p_uncond=p_uncond)
        assert input_adaptor is not None
        self.tandem_type = 'mel'

        self.melspec_loss = MelSpecReconstructionLoss(sample_rate=sample_rate)
        self.rvm = RelativeVolumeMel(sample_rate=sample_rate)

        self.validation_step_outputs = []
        self.automatic_optimization = False
        assert num_bands == backbone.num_bands

    def configure_optimizers(self):
        gen_params = [
            {"params": self.feature_extractor.parameters()},
            {"params": self.reflow.backbone.parameters()},
            {"params": self.reflow.head.parameters()},
            {"params": self.input_adaptor.parameters()},
        ]
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
            print("detected inf or nan values in gradients. not updating model parameters")
            optimizer.zero_grad()

    def get_log_spec(self, audio):
        S = self.reflow.head.get_spec(audio)
        target = torch.log(S.abs().clamp_min_(1e-6))
        return target

    def process_context(self, phone_info):
        if len(phone_info) == 4:
            return phone_info
        elif len(phone_info) == 6:
            # phone_info[4] = self.reflow.get_eq_norm_stft(phone_info[4])
            phone_info[4] = self.feature_extractor(phone_info[4])
        else:
            raise ValueError(f"Invalid phone_info, #fields {len(phone_info)}")
        return phone_info

    def training_step(self, batch, batch_idx, **kwargs):
        audio_input, phone_info = batch
        phone_info = self.process_context(phone_info)
        mel = self.feature_extractor(audio_input, **kwargs)
        tandem_feat = mel if self.tandem_type == 'mel' else self.get_log_spec(audio_input)
        # train generator
        opt_gen = self.optimizers()
        sch_gen = self.lr_schedulers()

        opt_gen.zero_grad()
        token_ids, tk_durs, tk_start, _, ctx, ctx_n_frame = phone_info
        tk_lens = (token_ids != 0).sum(1).long()
        tk_emb, ctx = self.input_adaptor(token_ids, ctx)
        cond = (tk_emb, tk_start, tk_lens, ctx, ctx_n_frame, torch.max(tk_durs.sum(1)))
        cond_ = [c.detach().clone() for c in cond]
        cond, bandwidth_id, (z_t, t, target) = self.reflow.get_train_tuple(cond, tandem_feat, audio_input)
        loss, loss_dict = self.reflow.compute_loss(z_t, t, target, cond, bandwidth_id=bandwidth_id)
        self.manual_backward(loss)
        self.skip_nan(opt_gen)
        self.clip_gradients(opt_gen, gradient_clip_val=1., gradient_clip_algorithm="norm")
        opt_gen.step()
        sch_gen.step()

        self.log("train/total_loss", loss, prog_bar=True, logger=False)
        if self.global_step % 1000 == 0 and self.global_rank == 0:
            with torch.no_grad():
                mel_hat_traj, audio_hat_traj = self.reflow.sample_ode(cond_, N=100, **kwargs)
            audio_hat = audio_hat_traj[-1]
            mel_hat = mel_hat_traj[-1]
            tandem_mel_loss = F.mse_loss(mel_hat, tandem_feat)
            mel_loss = F.mse_loss(self.feature_extractor(audio_hat), mel)
            self.logger.log_metrics(
                {"train/total_loss": loss, "train/mel_loss": mel_loss,
                 "train/tandem_mel_loss": tandem_mel_loss}, step=self.global_step)
            loss_dict = dict((f'train/{k}', v) for k, v in loss_dict.items())
            self.logger.log_metrics(loss_dict, step=self.global_step)
            rvm_loss = self.rvm(audio_hat.unsqueeze(1), audio_input.unsqueeze(1))
            for k, v in rvm_loss.items():
                self.logger.log_metrics({f"train/{k}": v}, step=self.global_step)
        return loss

    def validation_step(self, batch, batch_idx, **kwargs):
        audio_input, phone_info = batch
        phone_info = self.process_context(phone_info)
        with torch.no_grad():
            token_ids, dur, tk_start, _, ctx, ctx_n_frame = phone_info
            tk_lens = (token_ids != 0).sum(1).long()
            mel = self.feature_extractor(audio_input, **kwargs)
            tandem_feat = mel if self.tandem_type == "mel" else self.get_log_spec(audio_input)
            tk_emb, ctx = self.input_adaptor(token_ids, ctx)
            cond = (tk_emb, tk_start, tk_lens, ctx, ctx_n_frame, torch.max(dur.sum(1)))
            mel_hat_traj, audio_hat_traj = self.reflow.sample_ode(cond, N=100, **kwargs)
        audio_hat = audio_hat_traj[-1]
        mel_hat = mel_hat_traj[-1]

        tandem_mel_loss = F.mse_loss(mel_hat, tandem_feat)
        mel_loss = F.mse_loss(self.feature_extractor(audio_hat), mel)
        rvm_loss = self.rvm(audio_hat.unsqueeze(1), audio_input.unsqueeze(1))
        phase_loss = compute_phase_error(audio_hat, audio_input, self.reflow.head.get_spec)

        output = {
            "tandem_mel_loss": tandem_mel_loss,
            "mel_loss": mel_loss,
            "audio_input": audio_input[0],
            "audio_pred": audio_hat[0],
            "mel_pred": mel_hat[0],
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
                outputs[0]['audio_input'], outputs[0]['audio_pred'], outputs[0]['mel_pred'])
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

    def on_train_start(self, *args):
        if self.global_rank == 0:
            save_code(None, self.logger.save_dir)
