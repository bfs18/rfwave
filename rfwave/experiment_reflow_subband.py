import math
import wandb

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
import torchaudio

from torch import nn
from rfwave.feature_extractors import FeatureExtractor
from rfwave.heads import FourierHead
from rfwave.helpers import plot_spectrogram_to_numpy, save_figure_to_numpy
from rfwave.loss import MelSpecReconstructionLoss
from rfwave.models import Backbone
from rfwave.modules import safe_log10
from rfwave.multi_band_processor import MultiBandProcessor, PQMFProcessor, STFTProcessor
from rfwave.rvm import RelativeVolumeMel
from rfwave.lr_schedule import get_cosine_schedule_with_warmup
from rfwave.input import InputAdaptor, InputAdaptorProject
from rfwave.helpers import save_code
from rfwave.instantaneous_frequency import compute_phase_loss, compute_phase_error, compute_instantaneous_frequency
from rfwave.feature_weight import get_feature_weight, get_feature_weight2
from rfwave.logit_normal import LogitNormal
from rfwave.dit import DiTRFBackbone


class RectifiedFlow(nn.Module):
    def __init__(self, backbon: Backbone, head: FourierHead,
                 num_steps=10, feature_loss=False, wave=False, num_bands=8, p_uncond=0., guidance_scale=1.):
        super().__init__()
        self.backbone = torch.compile(backbon)
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
        assert self.backbone.output_channels == self.head.n_fft // self.num_bands + 2 * self.overlap
        assert self.prev_cond == self.backbone.prev_cond
        assert self.wave ^ self.stft_norm
        assert self.right_overlap >= 1  # at least one to deal with the last dimension of fft feature.
        # t_sampling = 'logit_normal' if isinstance(backbon, DiTRFBackbone) else 'uniform'
        t_sampling = 'uniform'
        self.t_dist = LogitNormal(mu=0., sigma=1.) if t_sampling == 'logit_normal' else None
        if self.stft_norm:
            self.stft_processor = STFTProcessor(self.head.n_fft + 2)
        if self.equalizer:
            self.eq_processor = PQMFProcessor(subbands=8, taps=124, cutoff_ratio=0.071)
        if self.feature_loss:
            self.register_buffer(
                "feature_weight", get_feature_weight2(self.head.n_fft, self.head.hop_length))

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

    def get_z0(self, mel, bandwidth_id):
        bandwidth_id = bandwidth_id[0]
        if self.wave:
            nf = mel.shape[2] if self.head.padding == "same" else (mel.shape[2] - 1)
            r = torch.randn([mel.shape[0], self.head.hop_length * nf], device=mel.device)
            rf = self.stft(r)
            z0 = self.get_subband(rf, bandwidth_id)
        else:
            r = torch.randn([mel.shape[0], self.head.n_fft + 2, mel.shape[2]], device=mel.device)
            z0 = self.get_subband(r, bandwidth_id)
        return z0

    def get_joint_z0(self, mel):
        if self.wave:
            nf = mel.shape[2] if self.head.padding == "same" else (mel.shape[2] - 1)
            r = torch.randn([mel.shape[0], self.head.hop_length * nf], device=mel.device)
            rf = self.stft(r)
            z0 = self.get_joint_subband(rf)
        else:
            r = torch.randn([mel.shape[0], self.head.n_fft + 2, mel.shape[2]], device=mel.device)
            z0 = self.get_joint_subband(r)
        z0 = z0.reshape(z0.size(0) * self.num_bands, z0.size(1) // self.num_bands, z0.size(2))
        return z0

    def get_eq_norm_stft(self, audio):
        if self.equalizer:
            audio = self.eq_processor.project_sample(audio.unsqueeze(1)).squeeze(1)
        S = self.stft(audio)
        if self.stft_norm:
            S = self.stft_processor.project_sample(S)
        return S

    def get_z1(self, audio, bandwidth_id):
        bandwidth_id = bandwidth_id[0]
        S = self.get_eq_norm_stft(audio)
        z1 = self.get_subband(S, bandwidth_id)
        if self.prev_cond:
            cond_band = self.get_subband(S, bandwidth_id - 1)
            if self.cond_mask_right_overlap:
                cond_band = self.mask_cond(cond_band)
        else:
            cond_band = None
        return z1, cond_band

    def get_joint_z1(self, audio):
        S = self.get_eq_norm_stft(audio)
        z1 = self.get_joint_subband(S)
        z1 = z1.reshape(z1.size(0) * self.num_bands, z1.size(1) // self.num_bands, z1.size(2))
        return z1

    def get_wave(self, x):
        if self.stft_norm:
            x = self.stft_processor.return_sample(x)
        x = self.istft(x)
        if self.equalizer:
            x = self.eq_processor.return_sample(x.unsqueeze(1)).squeeze(1)
        return x

    def sample_t(self, shape, device):
        if self.t_dist is not None:
            return self.t_dist.sample(shape).to(device)
        else:
            return torch.rand(shape, device=device)

    def get_train_tuple(self, mel, audio_input):
        if self.prev_cond or not self.parallel_uncond:
            t = self.sample_t((mel.size(0),), device=mel.device)
            bandwidth_id = torch.tile(torch.randint(0, self.num_bands, (), device=mel.device), (mel.size(0),))
            bandwidth_id = torch.ones([mel.shape[0]], dtype=torch.long, device=mel.device) * bandwidth_id
            z0 = self.get_z0(mel, bandwidth_id)
            z1, cond_band = self.get_z1(audio_input, bandwidth_id)
            mel = torch.cat([mel, cond_band], 1) if self.prev_cond else mel
        else:
            t = self.sample_t((mel.size(0),), device=mel.device).repeat_interleave(self.num_bands, 0)
            z0 = self.get_joint_z0(mel)
            z1 = self.get_joint_z1(audio_input)
            bandwidth_id = torch.tile(torch.arange(self.num_bands, device=mel.device), (mel.size(0),))
            mel = torch.repeat_interleave(mel, self.num_bands, 0)
        t_ = t.view(-1, 1, 1)
        z_t = t_ * z1 + (1. - t_) * z0
        target = z1 - z0
        return mel, bandwidth_id, (z_t, t, target)

    def get_pred(self, z_t, t, mel, bandwidth_id, encodec_bandwidth_id=None, **kwargs):
        if 'start' in kwargs and kwargs['start'] is not None:
            n_rpt = z_t.size(0) // kwargs['start'].size(0)
            start = torch.repeat_interleave(kwargs['start'], n_rpt, 0)
        else:
            start = None
        backbone_kwargs = {'start': start, 'encodec_bandwidth_id': encodec_bandwidth_id}
        pred = self.backbone(z_t, t, mel, bandwidth_id, **backbone_kwargs)
        return pred

    def get_ts(self, N):
        ts = torch.linspace(0., 1., N + 1)
        # if self.t_dist is not None:
        #     ts = self.t_dist.inv_cdf(ts)
        return ts

    @torch.no_grad()
    def sample_ode_subband(self, mel, band, bandwidth_id,
                           encodec_bandwidth_id=None, N=None, keep_traj=False, **kwargs):
        ### NOTE: Use Euler method to sample from the learned flow
        if N is None:
            N = self.N
        traj = []  # to store the trajectory
        if self.prev_cond or not self.parallel_uncond:
            assert band is not None
            assert bandwidth_id is not None
            # get z0 must be called before pre-processing mel
            z0 = self.get_z0(mel, bandwidth_id)
            mel = torch.cat([mel, band], 1) if self.prev_cond else mel
        else:
            assert band is None
            assert bandwidth_id is None
            bandwidth_id = torch.tile(torch.arange(self.num_bands, device=mel.device), (mel.size(0),))
            z0 = self.get_joint_z0(mel)  # get z0 must be called before pre-processing mel
            mel = torch.repeat_interleave(mel, self.num_bands, 0)

        ts = self.get_ts(N)
        z = z0.detach()
        fs = (z.size(0) // self.num_bands, z.size(1) * self.num_bands, z.size(2))
        ss = z.shape
        for i, t in enumerate(ts[:-1]):
            dt = ts[i + 1] - t
            t_ = torch.ones(z.size(0)) * t
            if self.cfg:
                mel_ = torch.cat([mel, torch.ones_like(mel) * mel.mean(dim=(0, 2), keepdim=True)], dim=0)
                (z_, t_, bandwidth_id_) = [torch.cat([v] * 2, dim=0) for v in (z, t_, bandwidth_id)]
                pred = self.get_pred(z_, t_.to(mel.device), mel_, bandwidth_id_, encodec_bandwidth_id, **kwargs)
                pred, uncond_pred = torch.chunk(pred, 2, dim=0)
                pred = uncond_pred + self.guidance_scale * (pred - uncond_pred)
            else:
                pred = self.get_pred(z, t_.to(mel.device), mel, bandwidth_id, encodec_bandwidth_id, **kwargs)
            if self.wave:
                if self.prev_cond or not self.parallel_uncond:
                    pred = self.place_subband(pred, bandwidth_id[0])
                    pred = self.stft(self.istft(pred))
                    pred = self.get_subband(pred, bandwidth_id[0])
                    z = z.detach() + pred * dt
                else:
                    pred = self.place_joint_subband(pred.reshape(fs))
                    pred = self.stft(self.istft(pred))
                    pred  = self.get_joint_subband(pred).reshape(ss)
                    z = z.detach() + pred * dt
            else:
                z = z.detach() + pred * dt
            if i == N - 1 or keep_traj:
                traj.append(z.detach())
        return traj

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

    def sample_ode(self, mel, encodec_bandwidth_id=None, N=None, keep_traj=False, **kwargs):
        traj = []
        if self.prev_cond or not self.parallel_uncond:
            band = mel.new_zeros((mel.shape[0], 2 * (self.num_bins + self.overlap), mel.shape[2]), device=mel.device)
            for i in range(self.num_bands):
                bandwidth_id = torch.ones([mel.shape[0]], dtype=torch.long, device=mel.device) * i
                traj_i = self.sample_ode_subband(
                    mel, band, bandwidth_id, encodec_bandwidth_id=encodec_bandwidth_id,
                    N=N, keep_traj=keep_traj, **kwargs)
                band = traj_i[-1]
                if self.prev_cond:
                    band = torch.zeros_like(band)
                elif self.cond_mask_right_overlap:
                    band = self.mask_cond(band)
                traj.append(traj_i)
            traj = self.combine_subbands(traj)
        else:
            traj_f = self.sample_ode_subband(
                mel, None, None,
                encodec_bandwidth_id=encodec_bandwidth_id, N=N, keep_traj=keep_traj, **kwargs)
            traj = [self.place_joint_subband(tt.reshape(tt.size(0) // self.num_bands, -1, tt.size(2)))
                    for tt in traj_f]
        return [self.get_wave(tt) for tt in traj]

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

    def compute_rf_loss(self, pred, target, bandwidth_id):
        if self.wave:
            if self.time_balance_loss:
                pred, target = self.time_balance_for_loss(pred, target)
            diff = pred - target
            diff = self._place_diff(diff, bandwidth_id)
            loss = self.istft(diff).pow(2.).mean()
        else:
            if self.feature_loss:
                # if self.stft_norm:
                #     pred = self.stft_processor.return_sample(pred)
                #     target = self.stft_processor.return_sample(target)
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

    def compute_loss(self, z_t, t, target, mel, bandwidth_id, encodec_bandwidth_id=None, **kwargs):
        if self.cfg and np.random.uniform() < self.p_uncond:
            mel = torch.ones_like(mel) * mel.mean(dim=(0, 2), keepdim=True)
        pred = self.get_pred(z_t, t, mel, bandwidth_id, encodec_bandwidth_id=encodec_bandwidth_id, **kwargs)
        pred, z_t, t, target = [v.float() for v in (pred, z_t, t, target)]
        stft_loss = self.compute_stft_loss(z_t, t, target, pred, bandwidth_id) if self.stft_loss else 0.
        phase_loss = self.compute_phase_loss(z_t, t, target, pred, bandwidth_id) if self.phase_loss else 0.
        overlap_loss = self.compute_overlap_loss(pred) if self.overlap_loss else 0.
        loss = self.compute_rf_loss(pred, target, bandwidth_id)
        loss_dict = {"loss": loss, "stft_loss": stft_loss,
                     "phase_loss": phase_loss, "overlap_loss": overlap_loss}
        return loss + (stft_loss + phase_loss + overlap_loss) * 0.1, loss_dict


class VocosExp(pl.LightningModule):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        backbone: Backbone,
        head: FourierHead,
        input_adaptor: InputAdaptor = None,
        task: str = "voc",
        sample_rate: int = 24000,
        initial_learning_rate: float = 2e-4,
        feature_loss: bool = False,
        wave: bool = False,
        num_bands: int = 8,
        guidance_scale: float = 1.,
        p_uncond: float = 0.2,
        num_warmup_steps: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["feature_extractor", "backbone", "head", "input_adaptor"])

        self.task = task
        self.feature_extractor = feature_extractor
        self.input_adaptor = input_adaptor
        self.reflow = RectifiedFlow(
            backbone, head, feature_loss=feature_loss, wave=wave, num_bands=num_bands,
            guidance_scale=guidance_scale, p_uncond=p_uncond)
        self.aux_loss = False
        self.aux_type = 'mel'
        if self.task == "tts":
            assert input_adaptor is not None
            self.input_adaptor = torch.compile(self.input_adaptor)
            if self.aux_loss:
                feat_dim = feature_extractor.dim if self.aux_type == 'mel' else self.reflow.head.n_fft // 2 + 1
                self.input_adaptor_proj = InputAdaptorProject(self.input_adaptor.dim, feat_dim)

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
        ]
        if self.input_adaptor is not None:
            gen_params.append({"params": self.input_adaptor.parameters()})
        if self.task == 'tts' and self.aux_loss:
            gen_params.append({"params": self.input_adaptor_proj.parameters()})

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

    def compute_aux_loss(self, features, audio_input, loss_type):
        if loss_type == 'mel':
            target = self.feature_extractor(audio_input)
        else:
            S = self.reflow.head.get_spec(audio_input)
            target = torch.log(S.abs().clamp_min_(1e-6))
        pred = self.input_adaptor_proj(features, torch.min(target))
        loss = F.mse_loss(pred, target)
        return loss

    def process_context(self, phone_info):
        if len(phone_info) == 4:
            return phone_info
        elif len(phone_info) == 6:
            # phone_info[4] = self.reflow.get_eq_norm_stft(phone_info[4])
            phone_info[4] = self.feature_extractor(phone_info[4])
        else:
            raise ValueError(f"Invalid phone_info, #fields {len(phone_info)}")
        return phone_info

    def on_before_optimizer_step(self, optimizer):
        # Note: `unscale` happens after the closure is executed, but before the `on_before_optimizer_step` hook.
        self.skip_nan(optimizer)
        self.clip_gradients(optimizer, gradient_clip_val=5., gradient_clip_algorithm="norm")

    def training_step(self, batch, batch_idx, **kwargs):
        if self.task == 'tts':
            audio_input, phone_info = batch
            phone_info = self.process_context(phone_info)
            start = None
        else:
            audio_input, audio_start = batch
            start = audio_start // self.reflow.head.hop_length
        mel = self.feature_extractor(audio_input, **kwargs)
        # train generator
        opt_gen = self.optimizers()
        sch_gen = self.lr_schedulers()
        opt_gen.zero_grad()

        if self.task == 'tts':
            features = self.input_adaptor(*phone_info)
            cond_mel_loss = self.compute_aux_loss(features, audio_input, self.aux_type) if self.aux_loss else 0.
        else:
            features = mel
            cond_mel_loss = 0.
        features_ext, bandwidth_id, (z_t, t, target) = self.reflow.get_train_tuple(features, audio_input)
        bi = kwargs.get("encodec_bandwidth_id", None)
        kwargs['start'] = start
        loss, loss_dict = self.reflow.compute_loss(
            z_t, t, target, features_ext, bandwidth_id=bandwidth_id, encodec_bandwidth_id=bi, start=start)
        loss = loss + cond_mel_loss
        self.manual_backward(loss)
        opt_gen.step()
        sch_gen.step()

        self.log("train/total_loss", loss, prog_bar=True, logger=False)
        if self.global_step % 1000 == 0 and self.global_rank == 0:
            with torch.no_grad():
                audio_hat_traj = self.reflow.sample_ode(features, N=100, **kwargs)
            audio_hat = audio_hat_traj[-1]
            mel_loss = F.mse_loss(self.feature_extractor(audio_hat, **kwargs), mel)
            self.logger.log_metrics(
                {"train/total_loss": loss, "train/cond_mel_loss": cond_mel_loss,
                 "train/mel_loss": mel_loss}, step=self.global_step)
            loss_dict = dict((f'train/{k}', v) for k, v in loss_dict.items())
            self.logger.log_metrics(loss_dict, step=self.global_step)
            rvm_loss = self.rvm(audio_hat.unsqueeze(1), audio_input.unsqueeze(1))
            for k, v in rvm_loss.items():
                self.logger.log_metrics({f"train/{k}": v}, step=self.global_step)

        return loss

    def validation_step(self, batch, batch_idx, **kwargs):
        if self.task == 'tts':
            audio_input, phone_info = batch
            phone_info, start = self.process_context(phone_info)
        else:
            audio_input, start = batch
        kwargs['start'] = None
        with torch.no_grad():
            features = self.feature_extractor(audio_input, **kwargs)
            cond = self.input_adaptor(*phone_info) if self.task == 'tts' else features
            audio_hat_traj = self.reflow.sample_ode(cond, N=100, **kwargs)
            cond_mel_hat = self.input_adaptor_proj(cond) if self.aux_loss and self.task == 'tts' else None
        audio_hat = audio_hat_traj[-1]

        mel_loss = F.mse_loss(self.feature_extractor(audio_hat, **kwargs), features)
        rvm_loss = self.rvm(audio_hat.unsqueeze(1), audio_input.unsqueeze(1))
        cond_mel_loss = (F.mse_loss(cond_mel_hat, features) if cond_mel_hat is not None
                         else torch.zeros(1, device=self.device))
        phase_loss = compute_phase_error(audio_hat, audio_input, self.reflow.head.get_spec)

        output = {
            "mel_loss": mel_loss,
            "audio_input": audio_input[0],
            "audio_pred": audio_hat[0],
            "cond_mel_loss": cond_mel_loss,
            "phase_loss": phase_loss,
        }
        if cond_mel_hat is not None:
            output['cond_mel_pred'] = cond_mel_hat[0]
        output.update(rvm_loss)
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self, **kwargs):
        outputs = self.validation_step_outputs
        mel_loss = torch.stack([x["mel_loss"] for x in outputs]).mean()
        cond_mel_loss = torch.stack([x["cond_mel_loss"] for x in outputs]).mean()
        phase_loss = torch.stack([x["phase_loss"] for x in outputs]).mean()
        rvm_loss_dict = {}
        for k in outputs[0].keys():
            if k.startswith("rvm"):
                rvm_loss_dict[f'valid/{k}'] = torch.stack([x[k] for x in outputs]).mean()

        self.log("val_loss", mel_loss, sync_dist=True, logger=False)
        if self.global_rank == 0:
            audio_in, audio_pred = outputs[0]['audio_input'].float(), outputs[0]['audio_pred'].float()
            mel_target = self.feature_extractor(audio_in.unsqueeze(0), **kwargs)[0]
            mel_hat = self.feature_extractor(audio_pred.unsqueeze(0), **kwargs)[0]
            metrics = {
                "valid/mel_loss": mel_loss,
                "valid/cond_mel_loss": cond_mel_loss,
                "valid/phase_loss": phase_loss}
            self.logger.log_metrics({**metrics, **rvm_loss_dict}, step=self.global_step)
            self.logger.experiment.log(
                {"valid_media/audio_in": wandb.Audio(audio_in.data.cpu().numpy(), sample_rate=self.hparams.sample_rate),
                 "valid_media/audio_hat": wandb.Audio(audio_pred.data.cpu().numpy(), sample_rate=self.hparams.sample_rate),
                 "valid_media/mel_in": wandb.Image(plot_spectrogram_to_numpy(mel_target.data.cpu().numpy())),
                 "valid_media/mel_hat": wandb.Image(plot_spectrogram_to_numpy(mel_hat.data.cpu().numpy()))},
                step=self.global_step)
            if 'cond_mel_pred' in outputs[0]:
                self.logger.experiment.log(
                    {"valid_media/cond_mel_hat": wandb.Image(
                        plot_spectrogram_to_numpy(outputs[0]['cond_mel_pred'].data.cpu().numpy()))},
                    step=self.global_step)
        self.validation_step_outputs.clear()

    # def on_train_epoch_start(self, *args):
    #     torch.cuda.empty_cache()

    def on_train_start(self, *args):
        if self.global_rank == 0:
            code_fp = save_code(None, self.logger.save_dir)
            # backup code to wandb
            artifact = wandb.Artifact(code_fp.stem, type='code')
            artifact.add_file(str(code_fp), name=code_fp.name)
            self.logger.experiment.log_artifact(artifact)


class VocosEncodecExp(VocosExp):
    """
    VocosEncodecExp is a subclass of VocosExp that overrides the parent experiment to function as a conditional GAN.
    It manages an additional `bandwidth_id` attribute, which denotes a learnable embedding corresponding to
    a specific bandwidth value of EnCodec. During training, a random bandwidth_id is generated for each step,
    while during validation, a fixed bandwidth_id is used.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        backbone: Backbone,
        head: FourierHead,
        sample_rate: int,
        initial_learning_rate: float,
        feature_loss: bool = False,
        wave: bool = False,
        num_bands: int = 8,
        guidance_scale: float = 1.,
        p_uncond: float = 0.2,
        num_warmup_steps: int = 0,
    ):
        super().__init__(
            feature_extractor=feature_extractor,
            backbone=backbone,
            head=head,
            input_adaptor=None,
            task="voc",
            sample_rate=sample_rate,
            initial_learning_rate=initial_learning_rate,
            feature_loss=feature_loss,
            wave=wave,
            num_bands=num_bands,
            guidance_scale=guidance_scale,
            p_uncond=p_uncond,
            num_warmup_steps=num_warmup_steps,
        )

    def training_step(self, *args):
        bandwidth_id = torch.randint(low=0, high=len(self.feature_extractor.bandwidths), size=(1,), device=self.device,)
        output = super().training_step(*args, encodec_bandwidth_id=bandwidth_id)
        return output

    def validation_step(self, *args):
        bandwidth_id = torch.randint(low=0, high=len(self.feature_extractor.bandwidths), size=(1,), device=self.device,)
        output = super().validation_step(*args, encodec_bandwidth_id=bandwidth_id)
        return output

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        bandwidth_id = torch.randint(
            low=0, high=len(self.feature_extractor.bandwidths), size=(1,), device=self.device)
        if self.global_rank == 0:
            audio_in = outputs[0]['audio_input']
            # Resynthesis with encodec for reference
            self.feature_extractor.encodec.set_target_bandwidth(self.feature_extractor.bandwidths[bandwidth_id])
            encodec_audio = self.feature_extractor.encodec(audio_in[None, None, :])
            self.logger.experiment.log(
                {"valid_media/encodec":
                     wandb.Audio(encodec_audio[0, 0].data.cpu().numpy(), sample_rate=self.hparams.sample_rate)},
                step=self.global_step-1 if self.global_step > 0 else 0)  # avoid a wired bug in wandb
        super().on_validation_epoch_end(encodec_bandwidth_id=bandwidth_id)


if __name__ == '__main__':
    import torchaudio
    from rfwave.feature_extractors import MelSpectrogramFeatures
    from rfwave.models import VocosRFBackbone
    from rfwave.heads import RFSTFTHead, RawFFTHead
    torch.set_printoptions(8)

    num_samples = 32512
    wav_fp = '../tests/biaobeifemale4us_average_00001.wav'
    wave = False
    feature_loss = False
    num_bands = 8

    torch.manual_seed(12345)
    torch.cuda.manual_seed(12345)
    feature_extractor = MelSpectrogramFeatures(
        sample_rate=24000, n_fft=1024, hop_length=256, n_mels=100, padding='center')
    backbone = VocosRFBackbone(
        input_channels=100, output_channels=160, dim=512, intermediate_dim=1536, num_layers=8, num_bands=8,
        with_fourier_features=False, encodec_num_embeddings=None, prev_cond=False)
    head = RFSTFTHead(dim=512, n_fft=1024, hop_length=256, padding='center')
    # head = RawFFTHead(n_fft=1024, hop_length=256)
    exp = VocosExp(
        feature_extractor=feature_extractor, backbone=backbone, head=head,
        wave=wave, feature_loss=feature_loss, num_bands=num_bands,
        sample_rate=24000, initial_learning_rate=2e-4)

    y, sr = torchaudio.load(wav_fp)
    y = y[:, :num_samples].requires_grad_(True)
    y = torch.repeat_interleave(y, 8, dim=0)
    features = exp.feature_extractor(y)
    cond, bandwidth_id, (z_t, t, target) = exp.reflow.get_train_tuple(features, y)

    z0 = z_t - t.view(-1, 1, 1) * target
    overlap_loss = exp.reflow.compute_overlap_loss(z0)
    print('overlap_loss', overlap_loss)  # should be 0.
    # print(z_t.shape, z_t[:, :10, :10])

    pred = exp.reflow.get_pred(z_t, t, cond, bandwidth_id)

    loss, _ = exp.reflow.compute_loss(z_t, t, target, cond, bandwidth_id)
    print(loss)
    print('grad norm', torch.norm(torch.autograd.grad(loss, z_t)[0]))

    y = exp.reflow.sample_ode(features)
