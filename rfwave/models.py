from typing import Optional, List, Union

import torch
import math
from torch import nn
from torch.nn.utils import weight_norm

from rfwave.modules import ConvNeXtV2Block, ResBlock1, AdaLayerNorm, GroupLinear, GroupLayerNorm


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


class VocosBackbone(Backbone):
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
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        adanorm_num_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.adanorm = adanorm_num_embeddings is not None
        if adanorm_num_embeddings:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.convnext = nn.ModuleList(
            [
                ConvNeXtV2Block(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    adanorm_num_embeddings=adanorm_num_embeddings,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, bandwidth_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embed(x)
        if self.adanorm:
            assert bandwidth_id is not None
            x = self.norm(x.transpose(1, 2), cond_embedding_id=bandwidth_id)
        else:
            x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x, cond_embedding_id=bandwidth_id)
        x = self.final_layer_norm(x.transpose(1, 2))
        return x


class VocosResNetBackbone(Backbone):
    """
    Vocos backbone module built with ResBlocks.

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        num_blocks (int): Number of ResBlock1 blocks.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to None.
    """

    def __init__(
        self, input_channels, dim, num_blocks, layer_scale_init_value=None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = weight_norm(nn.Conv1d(input_channels, dim, kernel_size=3, padding=1))
        layer_scale_init_value = layer_scale_init_value or 1 / num_blocks / 3
        self.resnet = nn.Sequential(
            *[ResBlock1(dim=dim, layer_scale_init_value=layer_scale_init_value) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.embed(x)
        x = self.resnet(x)
        x = x.transpose(1, 2)
        return x


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
        pred_var: Optional[bool] = False,
        pe_scale: float = 1000.,
        with_fourier_features: bool = True,
    ):
        super().__init__()
        self.pred_var = pred_var
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
        if pred_var:
            self.var_net = VocosBackbone(
                mel_ch, dim // 4, intermediate_dim // 4, num_layers // 2, None)
            self.var_out = nn.Linear(dim // 4, num_bands)
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

    def get_log_var(self, x):
        if self.pred_var:
            x = self.var_net(x)
            log_var = self.get_out(self.var_out, x)
            return log_var
        else:
            return None

    @staticmethod
    def get_out(out_layer, x):
        x = out_layer(x).transpose(1, 2)
        return x

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, x: torch.Tensor,
                bandwidth_id=None, encodec_bandwidth_id: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        pred_var: Optional[bool] = False,
        pe_scale: float = 1000.,
        with_fourier_features: bool = True,
    ):
        super().__init__()
        assert not pred_var, "pred_var should not be used for tandem tts."
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
            pred_var=pred_var,
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
            pred_var=pred_var,
            pe_scale=pe_scale,
            with_fourier_features=with_fourier_features)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, x: torch.Tensor, bandwidth_id: torch.Tensor):
        z_t1, z_t2 = torch.split(z_t, [self.output_channels1, self.output_channels2], dim=1)
        pred1 = self.module1(z_t1, t, x)
        est = z_t1 + (1 - t.view(-1, 1, 1)) * pred1
        cond2 = torch.cat([x, est], dim=1)
        pred2 = self.module2(z_t2, t, cond2, bandwidth_id)
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
        pred_var: Optional[bool] = False,
        pe_scale: float = 1000.,
        with_fourier_features: bool = True,
    ):
        super().__init__()
        assert not pred_var, "pred_var should not be used for tandem tts."
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
            pred_var=pred_var,
            pe_scale=pe_scale,
            with_fourier_features=with_fourier_features)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, x: torch.Tensor, bandwidth_id):
        return self.module(z_t, t, x, bandwidth_id)


class VocosRFGroupBackbone(Backbone):
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
        num_bands: int,
        band_multi: int = 2,
        share_main: bool = True,
        dilation: Union[int, List[int]] = 1,
        encodec_num_embeddings: Optional[int] = None,
        pred_var: Optional[bool] = False,
        pe_scale: float = 1000.,
        with_fourier_features: bool = True,
    ):
        super().__init__()
        self.pred_var = pred_var
        self.output_channels = output_channels
        self.joint_output_channels = output_channels * num_bands
        self.input_channels = input_channels
        self.with_fourier_features = with_fourier_features
        self.num_bands = num_bands
        self.band_multi = band_multi
        self.share_main = share_main
        band_dim = dim * self.band_multi
        band_intermediate_dim = dim * self.band_multi
        if self.share_main:
            main_dim = dim
            main_intermediate_dim = intermediate_dim
            num_main_bands = 1
        else:
            main_dim = dim * self.band_multi
            main_intermediate_dim = intermediate_dim * self.band_multi
            num_main_bands = num_bands
        # fourier module
        if self.with_fourier_features:
            self.fourier_module = Base2FourierFeatures(start=6, stop=8, step=1)
            fourier_dim = output_channels * 2 * (
                    (self.fourier_module.stop - self.fourier_module.start) // self.fourier_module.step)
        else:
            fourier_dim = 0
        # log var module
        if pred_var:
            self.var_net = VocosBackbone(input_channels, dim, intermediate_dim, num_layers, None)
            self.var_out = nn.Linear(dim, num_bands)
        # in band module
        self.embed = nn.Conv1d(
            (input_channels + fourier_dim + output_channels) * num_bands, band_dim,
            kernel_size=7, padding=3, groups=num_bands)
        self.adanorm = False
        self.norm = self.get_norm_layer(band_dim, None, gropus=num_bands)
        self.in_convnext = nn.ModuleList(
            [
                ConvNeXtV2Block(
                    dim=band_dim,
                    intermediate_dim=band_intermediate_dim,
                    adanorm_num_embeddings=None,
                    groups=num_bands,
                    dilation=dilation
                )
                for _ in range(num_layers // 2)
            ]
        )
        # in projection module
        self.in_proj = nn.Conv1d(band_dim, main_dim, kernel_size=7, padding=3, groups=num_main_bands)
        self.in_proj_norm = self.get_norm_layer(main_dim, None, gropus=num_main_bands)
        # main module
        self.convnext = nn.ModuleList(
            [
                ConvNeXtV2Block(
                    dim=main_dim,
                    intermediate_dim=main_intermediate_dim,
                    adanorm_num_embeddings=None,
                    groups=num_main_bands,
                    dilation=dilation
                )
                for _ in range(num_layers)
            ]
        )
        # main projection module
        self.proj = nn.Conv1d(main_dim, band_dim, kernel_size=7, padding=3, groups=num_main_bands)
        self.proj_norm = self.get_norm_layer(band_dim, None, gropus=num_main_bands)
        # out band module
        self.out_convnext = nn.ModuleList(
            [
                ConvNeXtV2Block(
                    dim=band_dim,
                    intermediate_dim=band_intermediate_dim,
                    adanorm_num_embeddings=None,
                    groups=num_bands,
                    dilation=dilation
                )
                for _ in range(num_layers // 2)
            ]
        )
        self.final_layer_norm = GroupLayerNorm(num_bands, band_dim, eps=1e-6)
        self.out = GroupLinear(band_dim, self.joint_output_channels, groups=num_bands)
        # time embedding
        self.pe_scale = pe_scale
        self.time_pos_emb = SinusoidalPosEmb(dim)
        t1_dim = main_dim // num_main_bands
        t2_dim = band_dim // num_bands
        bs = (num_main_bands, num_bands)
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4), nn.GELU(),
            torch.nn.Linear(dim * 4, t1_dim + t2_dim))
        # bandwidth embedding
        if encodec_num_embeddings is not None:
            self.encodec_bandwidth_emb = nn.Embedding(encodec_num_embeddings, t1_dim + t2_dim)
        else:
            self.encodec_bandwidth_emb = None
        self.apply(self._init_weights)
        def _split_time_emb(x):
            x_t1, x_t2 = torch.split(x, [t1_dim, t2_dim], dim=1)
            return torch.tile(x_t1, [1, bs[0], 1]), torch.tile(x_t2, [1, bs[1], 1])
        self.split_time_emb = _split_time_emb

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def get_norm_layer(self, dim, adanorm_num_embeddings, gropus):
        if adanorm_num_embeddings:
            norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        elif gropus == 1:
            norm = nn.LayerNorm(dim, eps=1e-6)
        else:
            norm = GroupLayerNorm(gropus, dim, eps=1e-6)
        return norm

    def apply_norm(self, norm_layer, x, bandwidth_id):
        if self.adanorm:
            assert bandwidth_id is not None
            x = norm_layer(x.transpose(1, 2), cond_embedding_id=bandwidth_id)
        else:
            x = norm_layer(x.transpose(1, 2))
        return x.transpose(1, 2)

    def get_log_var(self, x):
        if self.pred_var:
            x = self.var_net(x)
            log_var = self.get_out(self.var_out, x)
            return log_var
        else:
            return None

    @staticmethod
    def get_out(out_layer, x):
        x = out_layer(x).transpose(1, 2)
        return x

    def get_input(self, z_t, x):
        if self.with_fourier_features:
            z_t_f = self.fourier_module(z_t)
            z_t_b = torch.chunk(z_t, self.num_bands, dim=1)
            z_t_f_b = torch.chunk(z_t_f, self.num_bands, dim=1)
            x = torch.cat([torch.cat([zi, fi, x], dim=1)
                           for zi, fi in zip(z_t_b, z_t_f_b)], dim=1)
        else:
            z_t_b = torch.chunk(z_t, self.num_bands, dim=1)
            x = torch.cat([torch.cat([zi, x], dim=1) for zi in z_t_b], dim=1)
        return x

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, x: torch.Tensor,
                encodec_bandwidth_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.get_input(z_t, x)
        x = self.embed(x)
        x = self.apply_norm(self.norm, x, bandwidth_id=None)

        # embedding for time and bandwidth.
        emb_t = self.time_mlp(self.time_pos_emb(t, scale=self.pe_scale)).unsqueeze(2)
        main_emb_t, band_emb_t = self.split_time_emb(emb_t)
        if self.encodec_bandwidth_emb is not None:
            assert encodec_bandwidth_id is not None
            emb_b = self.encodec_bandwidth_emb(encodec_bandwidth_id).unsqueeze(-1)
            main_emb_b, band_emb_b = self.split_time_emb(emb_b)
        else:
            main_emb_b, band_emb_b = 0., 0.

        for conv_block in self.in_convnext:
            x = conv_block(x + band_emb_t + band_emb_b, cond_embedding_id=None)
        x = self.in_proj(x)
        x = self.apply_norm(self.in_proj_norm, x, None)
        for conv_block in self.convnext:
            x = conv_block(x + main_emb_t + main_emb_b, cond_embedding_id=None)
        x = self.proj(x)
        x = self.apply_norm(self.proj_norm, x, None)
        for conv_block in self.out_convnext:
            x = conv_block(x + band_emb_t + band_emb_b, cond_embedding_id=None)
        x = self.final_layer_norm(x.transpose(1, 2))
        x = self.get_out(self.out, x)
        return x


class VocosRFGroupTTSMultiTaskBackbone(Backbone):
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
            band_multi: int = 2,
            share_main: bool = True,
            dilation: Union[int, List[int]] = 1,
            pred_var: Optional[bool] = False,
            pe_scale: float = 1000.,
            with_fourier_features: bool = True,
    ):
        super().__init__()
        assert not pred_var, "pred_var should not be used for tandem tts."
        self.input_channels = input_channels
        self.output_channels1 = output_channels1
        self.output_channels2 = output_channels2
        self.num_bands = num_bands

        self.module = VocosRFGroupBackbone(
            input_channels=input_channels,
            output_channels=output_channels1 + output_channels2,
            dim=dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers1 + num_layers2,
            num_bands=num_bands,
            band_multi=band_multi,
            share_main=share_main,
            dilation=dilation,
            encodec_num_embeddings=None,
            pred_var=pred_var,
            pe_scale=pe_scale,
            with_fourier_features=with_fourier_features)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, x: torch.Tensor):
        out = self.module(z_t, t, x)
        return out
