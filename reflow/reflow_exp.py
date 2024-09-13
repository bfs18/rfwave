import copy
import torch
import pytorch_lightning as pl
import wandb
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict
from pathlib import Path

from rfwave.experiment_reflow_subband import RectifiedFlow
from rfwave.heads import FourierHead
from rfwave.models import Backbone
from rfwave.rvm import RelativeVolumeMel
from rfwave.lr_schedule import get_cosine_schedule_with_warmup
from rfwave.helpers import plot_spectrogram_to_numpy, save_code
from rfwave.instantaneous_frequency import compute_phase_error
from rfwave.feature_extractors import FeatureExtractor
from rfwave.modules import safe_log10


class Reflow(RectifiedFlow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stft_loss = True
        self.overlap_loss = True
        self.mel_loss = False

        if self.stft_norm:
            self.stft_processor.update = False
        if self.equalizer:
            self.eq_processor.update = False

    def get_train_tuple(self, batch):
        mel = batch['mel']
        z0 = batch['z0']
        z1 = batch['z1']
        bs = mel.size(0)

        t = torch.rand((bs,), device=mel.device).repeat_interleave(self.num_bands, 0)
        bandwidth_id = torch.tile(torch.arange(self.num_bands, device=mel.device), (bs,))

        z0 = self.get_joint_subband(self.stft(z0))
        z1 = self.get_joint_subband(self.get_eq_norm_stft(z1))
        z0, z1 = [x.reshape(x.size(0) * self.num_bands, x.size(1) // self.num_bands, x.size(2))
                  for x in [z0, z1]]
        mel = torch.repeat_interleave(mel, self.num_bands, 0)
        t_ = t.view(-1, 1, 1)
        z_t = t_ * z1 + (1. - t_) * z0
        target = z1 - z0
        return mel, bandwidth_id, (z_t, t, target)

    def get_one_step_train_tuple(self, batch):
        mel = batch['mel']
        z0 = batch['z0']
        z1 = batch['z1']
        bs = mel.size(0)

        # always start from 0.
        t = torch.zeros((bs,), device=mel.device).repeat_interleave(self.num_bands, 0)
        bandwidth_id = torch.tile(torch.arange(self.num_bands, device=mel.device), (bs,))

        z0 = self.get_joint_subband(self.stft(z0))
        z1 = self.get_joint_subband(self.get_eq_norm_stft(z1))
        z0, z1 = [
            x.reshape(x.size(0) * self.num_bands, x.size(1) // self.num_bands, x.size(2)) for x in [z0, z1]]

        mel = torch.repeat_interleave(mel, self.num_bands, 0)
        z_t = z0
        target = z1 - z0
        return mel, bandwidth_id, (z_t, t, target)

    def from_pretrained(self, pretrained_ckpt_path):
        print('Reflow initialization, this does not influence model restore after model initialization.')
        print(f'Loading pretrained model from {pretrained_ckpt_path}.')
        assert Path(pretrained_ckpt_path).exists()
        state_dict = torch.load(pretrained_ckpt_path, map_location=torch.device('cpu'))
        reflow_state_dict = OrderedDict()
        mel_fb = state_dict['state_dict']['feature_extractor.mel_spec.mel_scale.fb']
        self.register_buffer('mel_fb', mel_fb, persistent=False)
        for k, v in state_dict['state_dict'].items():
            if k.startswith('reflow.'):
                k = k.replace('reflow.', '')
                n, _ = next(self.backbone.named_parameters())
                if '_orig_mod.' in n and '_orig_mod.' not in k:
                    k = k.replace('backbone.', 'backbone._orig_mod.')
                elif '_orig_mod.' in k and '_orig_mod.' not in n:
                    k = k.replace('_orig_mod.', '')
                reflow_state_dict[k] = v
        for n, v in self.named_parameters():
            if n not in reflow_state_dict and 'convnext_adaptor' not in n:
                raise RuntimeError(f'{n} is not found in pretrained model.')
        self.load_state_dict(reflow_state_dict, strict=False)
        del state_dict

    def remove_image(self, pred):
        ss = pred.shape
        fs = (ss[0] // self.num_bands, ss[1] * self.num_bands, ss[2])
        pred = self.place_joint_subband(pred.reshape(fs))
        pred = self.stft(self.istft(pred))
        pred = self.get_joint_subband(pred).reshape(ss)
        return pred

    def compute_mel_loss(self, z_t, t, target, pred, bandwidth_id):
        def _mag(S):
            r, i = torch.chunk(S, 2, dim=1)
            return torch.sqrt(r ** 2 + i ** 2)

        z0 = z_t - t.view(-1, 1, 1) * target
        pred_z1 = self._place_diff(z0 + pred, bandwidth_id)
        target_z1 = self._place_diff(z0 + target, bandwidth_id)
        pred_mag = _mag(pred_z1)
        pred_mel = torch.matmul(pred_mag.transpose(-1, -2), self.mel_fb).transpose(-1, -2)
        target_mag = _mag(target_z1)
        target_mel = torch.matmul(target_mag.transpose(-1, -2), self.mel_fb).transpose(-1, -2)
        pred_log_mel = safe_log10(pred_mel)
        target_log_mel = safe_log10(target_mel)
        mag_loss = F.mse_loss(pred_log_mel, target_log_mel)
        converge_loss = (torch.norm(pred_mel - target_mel, p="fro") /
                         (torch.norm(target_mel, p="fro") + 1))
        return mag_loss + converge_loss

    def compute_loss(self, z_t, t, target, mel, bandwidth_id, encodec_bandwidth_id=None):
        if self.cfg and np.random.uniform() < self.p_uncond:
            mel = torch.ones_like(mel) * mel.mean(dim=(2,), keepdim=True)
        pred = self.get_pred(z_t, t, mel, bandwidth_id, encodec_bandwidth_id=encodec_bandwidth_id)
        mel_loss = self.compute_mel_loss(z_t, t, target, pred, bandwidth_id) if self.mel_loss else 0.
        stft_loss = self.compute_stft_loss(z_t, t, target, pred, bandwidth_id) if self.stft_loss else 0.
        phase_loss = self.compute_phase_loss(z_t, t, target, pred, bandwidth_id) if self.phase_loss else 0.
        overlap_loss = self.compute_overlap_loss(pred) if self.overlap_loss else 0.
        loss = self.compute_rf_loss(pred, target, bandwidth_id)
        loss_dict = {"loss": loss, "stft_loss": stft_loss, "mel_loss": mel_loss,
                     "phase_loss": phase_loss, "overlap_loss": overlap_loss}
        return loss + mel_loss + (stft_loss + phase_loss + overlap_loss) * 0.01, loss_dict

    def compute_teacher_loss(self, teacher_model, mel, bandwidth_id, encodec_bandwidth_id=None):
        # this mel is repeated.
        z0 = self.get_joint_z0(mel.reshape(mel.shape[0] // self.num_bands, self.num_bands, *mel.shape[1:])[:, 0])
        if self.cfg and np.random.uniform() < self.p_uncond:
            mel = torch.ones_like(mel) * mel.mean(dim=(2,), keepdim=True)
        t0 = torch.zeros((z0.size(0),), device=z0.device)
        pred = self.get_pred(z0, t0, mel, bandwidth_id, encodec_bandwidth_id=encodec_bandwidth_id)
        # pred = teacher_model.get_pred(z0, t0, mel, bandwidth_id, encodec_bandwidth_id=encodec_bandwidth_id)
        with torch.no_grad():
            # from SlimFlow code
            t_ = torch.rand((mel.shape[0] // self.num_bands,), device=mel.device) * 0.6 + 0.2     # 0.2 ~ 0.8
            t_ = torch.repeat_interleave(t_, self.num_bands, dim=0)
            step_t = torch.einsum('b,bij->bij', t_, self.remove_image(pred))
            x_t_psuedo = z0 + step_t
            pred_teacher = teacher_model.get_pred(
                x_t_psuedo, t_, mel, bandwidth_id, encodec_bandwidth_id=encodec_bandwidth_id)
            step_1_t = torch.einsum('b,bij->bij', 1 - t_, self.remove_image(pred_teacher))
            pred_teacher = step_t + step_1_t
        loss = self.compute_rf_loss(pred, pred_teacher.detach(), bandwidth_id)
        mel_loss = self.compute_mel_loss(z0, t0, pred_teacher.detach(), pred, bandwidth_id) if self.mel_loss else 0.
        stft_loss = self.compute_stft_loss(z0, t0, pred_teacher, pred, bandwidth_id) if self.stft_loss else 0.
        overlap_loss = self.compute_overlap_loss(pred) if self.overlap_loss else 0.
        loss_dict = {'teacher_loss': loss, 'teacher_stft_loss': stft_loss, 'teacher_mel_loss': mel_loss,
                     'teacher_overlap_loss': overlap_loss, 'mel_loss': mel_loss}
        return loss + mel_loss + (stft_loss + overlap_loss) * 0.01, loss_dict

    def sample_teacher(self, teacher_model, mel, encodec_bandwidth_id=None):
        with torch.no_grad():
            bandwidth_id = torch.tile(torch.arange(self.num_bands, device=mel.device), (mel.size(0),))
            z0 = self.get_joint_z0(mel)
            t0 = torch.zeros((z0.size(0),), device=mel.device)
            mel = mel.repeat_interleave(self.num_bands, 0)
            pred = self.get_pred(z0, t0, mel, bandwidth_id, encodec_bandwidth_id=encodec_bandwidth_id)
            # pred = teacher_model.get_pred(z0, t0, mel, bandwidth_id, encodec_bandwidth_id=encodec_bandwidth_id)
            t_ = torch.rand((mel.shape[0] // self.num_bands,), device=mel.device) * 0.6 + 0.2     # 0.2 ~ 0.8
            t_ = torch.repeat_interleave(t_, self.num_bands, dim=0)
            step_t = torch.einsum('b,bij->bij', t_, self.remove_image(pred))
            x_t_psuedo = z0 + step_t
            pred_teacher = teacher_model.get_pred(
                x_t_psuedo, t_, mel, bandwidth_id, encodec_bandwidth_id=encodec_bandwidth_id)
            step_1_t = torch.einsum('b,bij->bij', 1 - t_, self.remove_image(pred_teacher))
            pred_teacher = step_t + step_1_t
            z1 = z0 + pred_teacher
        z1 = self.place_joint_subband(z1.reshape(z1.size(0) // self.num_bands, -1, z1.size(2)))
        wave = self.get_wave(z1)
        return wave

class ReflowExp(pl.LightningModule):
    def __init__(
            self,
            feature_extractor: FeatureExtractor,
            backbone: Backbone,
            head: FourierHead,
            pretrained_ckpt_path: str,
            one_step: bool = False,
            teacher: bool = False,
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
        self.save_hyperparameters(ignore=["feature_extractor", "backbone", "head"])

        self.feature_extractor = feature_extractor
        backbone = torch.compile(backbone)
        self.one_step = one_step
        self.teacher = teacher
        self.task = task
        self.reflow = Reflow(
            backbone, head, feature_loss=feature_loss, wave=wave, num_bands=num_bands,
            guidance_scale=guidance_scale, p_uncond=p_uncond)
        self.rvm = RelativeVolumeMel(sample_rate=sample_rate)

        self.validation_step_outputs = []
        self.automatic_optimization = False
        assert num_bands == backbone.num_bands

        self.reflow.from_pretrained(pretrained_ckpt_path)

        if self.one_step and self.teacher:
            self.teacher_reflow = copy.deepcopy(self.reflow)
            self.teacher_reflow.requires_grad_(False)
            self.teacher_reflow.eval()

    def configure_optimizers(self):
        gen_params = [
            {"params": self.reflow.backbone.parameters()},
            {"params": self.reflow.head.parameters()},
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

    def on_before_optimizer_step(self, optimizer):
        # Note: `unscale` happens after the closure is executed, but before the `on_before_optimizer_step` hook.
        self.skip_nan(optimizer)
        self.clip_gradients(optimizer, gradient_clip_val=5., gradient_clip_algorithm="norm")

    def training_step(self, batch, batch_idx, **kwargs):
        # train generator
        opt_gen = self.optimizers()
        sch_gen = self.lr_schedulers()
        opt_gen.zero_grad()

        if self.one_step:
            features_ext, bandwidth_id, (z_t, t, target) = self.reflow.get_one_step_train_tuple(batch)
        else:
            features_ext, bandwidth_id, (z_t, t, target) = self.reflow.get_train_tuple(batch)

        bi = kwargs.get("encodec_bandwidth_id", None)
        loss, loss_dict = self.reflow.compute_loss(
            z_t, t, target, features_ext, bandwidth_id=bandwidth_id, encodec_bandwidth_id=bi)
        if self.one_step and self.teacher:
            teacher_loss, teacher_loss_dict = self.reflow.compute_teacher_loss(
                self.teacher_reflow, features_ext, bandwidth_id=bandwidth_id, encodec_bandwidth_id=bi)
            loss = loss + teacher_loss
            loss_dict.update(**teacher_loss_dict)
        self.manual_backward(loss)
        opt_gen.step()
        sch_gen.step()

        self.log("train/total_loss", loss, prog_bar=True, logger=False)
        if self.global_step % 1000 == 0 and self.global_rank == 0:
            with torch.no_grad():
                N = 1 if self.one_step else 100
                audio_hat_traj = self.reflow.sample_ode(batch['mel'], N=N, **kwargs)
            audio_hat = audio_hat_traj[-1]
            mel_loss = F.mse_loss(self.feature_extractor(audio_hat, **kwargs), batch['mel'])
            self.logger.log_metrics(
                {"train/total_loss": loss, "train/mel_loss": mel_loss}, step=self.global_step)
            loss_dict = dict((f'train/{k}', v) for k, v in loss_dict.items())
            self.logger.log_metrics(loss_dict, step=self.global_step)
            rvm_loss = self.rvm(audio_hat.unsqueeze(1), batch['z1'].unsqueeze(1))
            for k, v in rvm_loss.items():
                self.logger.log_metrics({f"train/{k}": v}, step=self.global_step)
        return loss

    def validation_step(self, batch, batch_idx, **kwargs):
        with torch.no_grad():
            N = 1 if self.one_step else 100
            audio_hat_traj = self.reflow.sample_ode(batch['mel'], N=N, **kwargs)
        audio_hat = audio_hat_traj[-1]
        mel_loss = F.mse_loss(self.feature_extractor(audio_hat, **kwargs), batch['mel'])
        rvm_loss = self.rvm(audio_hat.unsqueeze(1), batch['z1'].unsqueeze(1))
        phase_loss = compute_phase_error(audio_hat, batch['z1'], self.reflow.head.get_spec)

        output = {
            "mel_loss": mel_loss,
            "audio_input": batch['z1'][0],
            "audio_pred": audio_hat[0],
            "phase_loss": phase_loss,
        }
        output.update(rvm_loss)

        if self.teacher:
            teacher_audio_hat = self.reflow.sample_teacher(self.teacher_reflow, batch['mel'], **kwargs)
            output['teacher_pred'] = teacher_audio_hat[0]

        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self, **kwargs):
        outputs = self.validation_step_outputs
        mel_loss = torch.stack([x["mel_loss"] for x in outputs]).mean()
        phase_loss = torch.stack([x["phase_loss"] for x in outputs]).mean()
        rvm_loss_dict = {}
        for k in outputs[0].keys():
            if k.startswith("rvm"):
                rvm_loss_dict[f'valid/{k}'] = torch.stack([x[k] for x in outputs]).mean()

        self.log("val_loss", mel_loss, sync_dist=True, logger=False)
        if self.global_rank == 0:
            audio_in, audio_pred = outputs[0]['audio_input'], outputs[0]['audio_pred']
            mel_target = self.feature_extractor(audio_in.unsqueeze(0), **kwargs)[0]
            mel_hat = self.feature_extractor(audio_pred.unsqueeze(0), **kwargs)[0]
            metrics = {
                "valid/mel_loss": mel_loss,
                "valid/phase_loss": phase_loss}
            self.logger.log_metrics({**metrics, **rvm_loss_dict}, step=self.global_step)
            media = {
                "valid_media/audio_in": wandb.Audio(
                    audio_in.float().cpu().numpy(), sample_rate=self.hparams.sample_rate),
                 "valid_media/audio_hat": wandb.Audio(
                     audio_pred.float().cpu().numpy(), sample_rate=self.hparams.sample_rate),
                 "valid_media/mel_in": wandb.Image(plot_spectrogram_to_numpy(mel_target.data.cpu().numpy())),
                 "valid_media/mel_hat": wandb.Image(plot_spectrogram_to_numpy(mel_hat.data.cpu().numpy()))}
            if self.teacher:
                audio_teacher = outputs[0]['teacher_pred']
                mel_teacher = self.feature_extractor(audio_teacher.unsqueeze(0), **kwargs)[0]
                media['valid_media/audio_teacher'] = wandb.Audio(
                    audio_teacher.float().cpu().numpy(), sample_rate=self.hparams.sample_rate)
                media['valid_media/mel_teacher'] = wandb.Image(
                    plot_spectrogram_to_numpy(mel_teacher.data.cpu().numpy()))
            self.logger.experiment.log(media, step=self.global_step)
        self.validation_step_outputs.clear()

    def on_train_start(self, *args):
        if self.global_rank == 0:
            save_code(None, self.logger.save_dir)
