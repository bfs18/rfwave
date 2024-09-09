import torch
import pytorch_lightning as pl
import wandb
import torch.nn.functional as F

from rfwave.experiment_reflow_subband import RectifiedFlow
from rfwave.heads import FourierHead
from rfwave.models import Backbone
from rfwave.rvm import RelativeVolumeMel
from rfwave.lr_schedule import get_cosine_schedule_with_warmup
from rfwave.helpers import plot_spectrogram_to_numpy, save_code
from rfwave.instantaneous_frequency import compute_phase_error


class Reflow(RectifiedFlow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_train_tuple(self, batch):
        mel = batch['mel']
        z0 = batch['z0']
        z1 = batch['z1']
        z0 = self.get_joint_subband(self.stft(z0))
        z1 = self.get_joint_subband(self.get_eq_norm_stft(z1))
        z0, z1 = [x.reshape(x.size(0) * self.num_bands, x.size(0) // self.num_bands, x.size(2))
                  for x in [z0, z1]]
        mel = torch.repeat_interleave(mel, self.num_bands, 0)
        bandwidth_id = torch.tile(torch.arange(self.num_bands, device=mel.device), (mel.size(0),))
        t = torch.rand((mel.size(0),), device=mel.device).repeat_interleave(self.num_bands, 0)
        t_ = t.view(-1, 1, 1)
        z_t = t_ * z1 + (1. - t_) * z0
        target = z1 - z0
        return mel, bandwidth_id, (z_t, t, target)


class ReflowExp(pl.LightningModule):
    def __init__(
            self,
            backbone: Backbone,
            head: FourierHead,
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
        self.save_hyperparameters(ignore=["backbone", "head"])

        backbone = torch.compile(backbone)
        self.task = task
        self.reflow = Reflow(
            backbone, head, feature_loss=feature_loss, wave=wave, num_bands=num_bands,
            guidance_scale=guidance_scale, p_uncond=p_uncond)
        self.rvm = RelativeVolumeMel(sample_rate=sample_rate)

        self.validation_step_outputs = []
        self.automatic_optimization = False
        assert num_bands == backbone.num_bands

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

    def training_step(self, batch, batch_idx, **kwargs):
        # train generator
        opt_gen = self.optimizers()
        sch_gen = self.lr_schedulers()
        opt_gen.zero_grad()

        features_ext, bandwidth_id, (z_t, t, target) = self.reflow.get_train_tuple(batch)
        bi = kwargs.get("encodec_bandwidth_id", None)
        loss, loss_dict = self.reflow.compute_loss(
            z_t, t, target, features_ext, bandwidth_id=bandwidth_id, encodec_bandwidth_id=bi)
        self.manual_backward(loss)
        self.skip_nan(opt_gen)
        opt_gen.step()
        sch_gen.step()

        self.log("train/total_loss", loss, prog_bar=True, logger=False)
        if self.global_step % 1000 == 0 and self.global_rank == 0:
            with torch.no_grad():
                audio_hat_traj = self.reflow.sample_ode(batch['mel'], N=100, **kwargs)
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
        if self.task == 'tts':
            audio_input, phone_info = batch
            phone_info = self.process_context(phone_info)
        else:
            audio_input = batch
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
            audio_in, audio_pred = outputs[0]['audio_input'], outputs[0]['audio_pred']
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

    def on_train_start(self, *args):
        if self.global_rank == 0:
            save_code(None, self.logger.save_dir)
