import math
import wandb

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
import torchaudio

from torch import nn
from rfwave.multi_band_processor import STFTProcessor
from rfwave.input import InputAdaptor
from rfwave.models import Backbone
from rfwave.lr_schedule import get_cosine_schedule_with_warmup
from rfwave.logit_normal import LogitNormal
from rfwave.helpers import save_code
from rfwave.dit import DiTRFBackbone


class RectifiedFlow(nn.Module):
    def __init__(self, backbone, num_steps=10., p_uncond=0., guidance_scale=1.):
        super().__init__()
        self.backbone = backbone
        self.N = num_steps
        self.cfg = guidance_scale > 1.
        self.p_uncond = p_uncond
        self.guidance_scale = guidance_scale
        self.dur_processor = STFTProcessor(1)
        self.validation_step_outputs = []
        self.automatic_optimization = False
        # t_sampling = 'logit_normal' if isinstance(backbone, DiTRFBackbone) else 'uniform'
        t_sampling = 'uniform'
        self.t_dist = LogitNormal(mu=0., sigma=1.) if t_sampling == 'logit_normal' else None

    def get_z0(self, text):
        return torch.randn(text.size(0), 1, text.size(2), device=text.device)

    def get_z1(self, dur):
        return self.dur_processor.project_sample(dur)

    def sample_t(self, shape, device):
        if self.t_dist is not None:
            return self.t_dist.sample(shape).to(device)
        else:
            return torch.rand(shape, device=device)

    def get_train_tuple(self, text, dur):
        z0 = self.get_z0(text)
        z1 = self.get_z1(dur)
        t = self.sample_t((z1.size(0),), device=z1.device)
        t_ = t.view(-1, 1, 1)
        z_t = t_ * z1 + (1. - t_) * z0
        target = z1 - z0
        return z_t, t, target

    def get_pred(self, z_t, t, text):
        pred = self.backbone(z_t, t, text)
        return pred

    @torch.no_grad()
    def sample_ode(self, text, N=None, keep_traj=False):
        if N is None:
            N = self.N
        dt = 1. / N
        z0 = self.get_z0(text)
        z = z0.detach()
        batchsize = z.shape[0]
        traj = []  # to store the trajectory
        for i in range(N):
            t = torch.ones((batchsize)) * i / N
            if self.cfg:
                text_ = torch.cat([text, torch.ones_like(text) * text.mean(dim=(0, 2), keepdim=True)], dim=0)
                (z_, t_) = [torch.cat([v] * 2, dim=0) for v in (z, t)]
                pred = self.get_pred(z_, t_.to(text.device), text_)
                pred, uncond_pred = torch.chunk(pred, 2, dim=0)
                pred = uncond_pred + self.guidance_scale * (pred - uncond_pred)
            else:
                pred = self.get_pred(z, t.to(text.device), text)
            z = z.detach() + pred * dt
            if i == N - 1 or keep_traj:
                traj.append(self.dur_processor.return_sample(z.detach()))
        return traj

    def compute_loss(self, z_t, t, target, text):
        if self.cfg and np.random.uniform() < self.p_uncond:
            text = torch.ones_like(text) * text.mean(dim=(0, 2), keepdim=True)
        pred = self.get_pred(z_t, t, text)
        pred, z_t, t, target = [v.float() for v in (pred, z_t, t, target)]
        mask = (text.abs().sum(1, keepdim=True) > 0.).float()
        loss = ((pred - target) ** 2 * mask).sum() / mask.sum()
        return loss


class VocosExp(pl.LightningModule):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        backbone: Backbone,
        input_adaptor: InputAdaptor = None,
        task: str = "dur",
        initial_learning_rate: float = 2e-4,
        guidance_scale: float = 1.,
        p_uncond: float = 0.2,
        num_warmup_steps: int = 0,
        torch_compile: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone", "input_adaptor"])
        self.task = task
        if torch_compile:
            input_adaptor = torch.compile(input_adaptor)
            backbone = torch.compile(backbone)
        self.input_adaptor = input_adaptor
        self.reflow = RectifiedFlow(backbone, p_uncond=p_uncond, guidance_scale=guidance_scale)
        self.validation_step_outputs = []
        self.automatic_optimization = False

    def configure_optimizers(self):
        gen_params = [
            {"params": self.reflow.backbone.parameters()},
            {"params": self.input_adaptor.parameters()}
        ]
        opt_gen = torch.optim.AdamW(gen_params, lr=self.hparams.initial_learning_rate)
        max_steps = self.trainer.max_steps
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
        token_ids, durs = batch
        opt_gen = self.optimizers()
        sch_gen = self.lr_schedulers()
        opt_gen.zero_grad()
        features = self.input_adaptor(token_ids)
        z_t, t, target = self.reflow.get_train_tuple(features, durs)
        loss = self.reflow.compute_loss(z_t, t, target, features)
        self.manual_backward(loss)
        opt_gen.step()
        sch_gen.step()
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx, **kwargs):
        token_ids, durs = batch
        with torch.no_grad():
            features = self.input_adaptor(token_ids)
            dur_traj = self.reflow.sample_ode(features, N=100, **kwargs)
        dur_hat = dur_traj[-1]
        mask = (token_ids != 0).float()
        loss = ((dur_hat - durs) ** 2 * mask.unsqueeze(1)).sum() / mask.sum()
        output = {"val_loss": loss}
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True, logger=True)
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
