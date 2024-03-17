from typing import Optional, List, Union

import torch
import math
from torch import nn
from torch.nn.utils import weight_norm

from rfwave.modules import ConvNeXtV2Block, ResBlock1, AdaLayerNorm


class Backbone(nn.Module):
    """Base class for the generator's backbone. It preserves the same temporal resolution across all layers."""

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, L), where B is the batch size,
                        C denotes output features, and L is the sequence length.

        Returns:
            Tensor: Output of shape (B, L, H), where B is the batch size, L is the sequence length,
                    and H denotes the model dimension.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Base2FourierFeatures(nn.Module):
    def __init__(self, start=0, stop=8, step=1):
        super().__init__()
        self.start = start
        self.stop = stop
        self.step = step

    def __call__(self, inputs):
        freqs = range(self.start, self.stop, self.step)

        # Create Base 2 Fourier features
        w = 2. ** (torch.tensor(freqs, dtype=inputs.dtype)).to(inputs.device) * 2 * torch.pi
        w = torch.tile(w[None, :, None], (1, inputs.shape[1], 1))

        # Compute features
        h = torch.repeat_interleave(inputs, len(freqs), dim=1)
        h = w * h
        h = torch.stack([torch.sin(h), torch.cos(h)], dim=2)
        return h.reshape(h.size(0), -1, h.size(3))


class VocosRFBackbone(Backbone):
    """
    Vocos backbone module built with ConvNeXt blocks. Supports additional conditioning with Adaptive Layer Normalization

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        intermediate_dim (int): Intermediate dimension used in ConvNeXtBlock.
        num_layers (int): Number of ConvNeXtBlock layers.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to `1 / num_layers`.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
                                                None means non-conditional model. Defaults to None.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        num_bands: Optional[int],
        dilation: Union[int, List[int]] = 1,
        encodec_num_embeddings: Optional[int] = None,
        prev_cond: Optional[bool] = True,
        pe_scale: float = 1000.,
        with_fourier_features: bool = True,
    ):
        super().__init__()
        self.prev_cond = prev_cond
        self.output_channels = output_channels
        self.with_fourier_features = with_fourier_features
        self.num_bands = num_bands
        if self.with_fourier_features:
            self.fourier_module = Base2FourierFeatures(start=6, stop=8, step=1)
            fourier_dim = output_channels * 2 * (
                    (self.fourier_module.stop - self.fourier_module.start) // self.fourier_module.step)
        else:
            fourier_dim = 0
        mel_ch = input_channels
        input_channels = mel_ch + output_channels if prev_cond else mel_ch
        self.input_channels = mel_ch
        self.embed = nn.Conv1d(input_channels + output_channels + fourier_dim, dim, kernel_size=7, padding=3)
        self.adanorm = num_bands is not None and num_bands > 1
        if self.adanorm:
            self.norm = AdaLayerNorm(num_bands, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        if isinstance(dilation, (list, tuple)):
            assert num_layers % len(dilation) == 0, "num_layers must be divisible by len(dilation) for cycled dilation"
            dilation_cycles = dilation * (num_layers // len(dilation))
        else:
            assert dilation is None or isinstance(dilation, int), "dilation must be an int or a list of ints"
            dilation_cycles = [dilation] * num_layers  # None also in this case.
        self.convnext = nn.ModuleList(
            [
                ConvNeXtV2Block(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    adanorm_num_embeddings=num_bands,
                    dilation=dilation_cycles[i],
                )
                for i in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.pe_scale = pe_scale
        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4), nn.GELU(), torch.nn.Linear(dim * 4, dim))
        if encodec_num_embeddings is not None:
            self.encodec_bandwidth_emb = nn.Embedding(encodec_num_embeddings, dim)
        else:
            self.encodec_bandwidth_emb = None
        self.out = nn.Linear(dim, output_channels)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    @staticmethod
    def get_out(out_layer, x):
        x = out_layer(x).transpose(1, 2)
        return x

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, x: torch.Tensor, bandwidth_id=None,
                encodec_bandwidth_id: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if self.with_fourier_features:
            z_t_f = self.fourier_module(z_t)
            x = self.embed(torch.cat([z_t, x, z_t_f], dim=1))
        else:
            x = self.embed(torch.cat([z_t, x], dim=1))
        emb_t = self.time_mlp(self.time_pos_emb(t, scale=self.pe_scale)).unsqueeze(2)
        if self.encodec_bandwidth_emb is not None:
            assert encodec_bandwidth_id is not None
            emb_b = self.encodec_bandwidth_emb(encodec_bandwidth_id).unsqueeze(-1)
        else:
            emb_b = 0.
        if self.adanorm:
            assert bandwidth_id is not None
            x = self.norm(x.transpose(1, 2), cond_embedding_id=bandwidth_id)
        else:
            x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x + emb_t + emb_b, cond_embedding_id=bandwidth_id)
        x = self.final_layer_norm(x.transpose(1, 2))
        x = self.get_out(self.out, x)
        return x


class VocosRFTTSTandemBackbone(Backbone):
    """
    Tandem TTS

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        intermediate_dim (int): Intermediate dimension used in ConvNeXtBlock.
        num_layers (int): Number of ConvNeXtBlock layers.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to `1 / num_layers`.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
                                                None means non-conditional model. Defaults to None.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels1: int,
        output_channels2: int,
        dim: int,
        intermediate_dim: int,
        num_layers1: int,
        num_layers2: int,
        num_bands: int,
        dilation: Union[int, List[int]] = 1,
        prev_cond: Optional[bool] = True,
        pe_scale: float = 1000.,
        with_fourier_features: bool = True,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels1 = output_channels1
        self.output_channels2 = output_channels2
        self.p_uncond = 0.1
        self.guidance_scale = 2.
        self.num_bands = num_bands

        self.module1 = VocosRFBackbone(
            input_channels=input_channels,
            output_channels=output_channels1,
            dim=dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers1,
            num_bands=None,
            dilation=dilation,
            encodec_num_embeddings=None,
            prev_cond=False,
            pe_scale=pe_scale,
            with_fourier_features=with_fourier_features)

        self.module2 = VocosRFBackbone(
            input_channels=output_channels1 + input_channels,
            output_channels=output_channels2,
            dim=dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers2,
            num_bands=num_bands,
            dilation=dilation,
            encodec_num_embeddings=None,
            prev_cond=prev_cond,
            pe_scale=pe_scale,
            with_fourier_features=with_fourier_features)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, x: torch.Tensor, bandwidth_id: torch.Tensor, **kwargs):
        z_t1, z_t2 = torch.split(z_t, [self.output_channels1, self.output_channels2], dim=1)
        pred1 = self.module1(z_t1, t, x, **kwargs)
        est = z_t1 + (1 - t.view(-1, 1, 1)) * pred1
        cond2 = torch.cat([x, est], dim=1)
        pred2 = self.module2(z_t2, t, cond2, bandwidth_id, **kwargs)
        return torch.cat([pred1, pred2], dim=1)


class VocosRFTTSMultiTaskBackbone(Backbone):
    """
    Tandem TTS

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        intermediate_dim (int): Intermediate dimension used in ConvNeXtBlock.
        num_layers (int): Number of ConvNeXtBlock layers.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to `1 / num_layers`.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
                                                None means non-conditional model. Defaults to None.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels1: int,
        output_channels2: int,
        dim: int,
        intermediate_dim: int,
        num_layers1: int,
        num_layers2: int,
        num_bands: int,
        dilation: Union[int, List[int]] = 1,
        prev_cond: Optional[bool] = True,
        pe_scale: float = 1000.,
        with_fourier_features: bool = True,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels1 = output_channels1
        self.output_channels2 = output_channels2
        self.num_bands = num_bands

        self.module = VocosRFBackbone(
            input_channels=input_channels,
            output_channels=output_channels1 + output_channels2,
            dim=dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers1 + num_layers2,
            num_bands=num_bands,
            dilation=dilation,
            encodec_num_embeddings=None,
            prev_cond=prev_cond,
            pe_scale=pe_scale,
            with_fourier_features=with_fourier_features)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, x: torch.Tensor, bandwidth_id, **kwargs):
        return self.module(z_t, t, x, bandwidth_id, **kwargs)
