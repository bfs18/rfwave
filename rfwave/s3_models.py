import torch

from torch import nn
from typing import Optional, List, Union
from rfwave.modules import ConvNeXtV2Block, AdaLayerNorm
from rfwave.models import Backbone, Base2FourierFeatures, SinusoidalPosEmb
from rfwave.input import Attention, ModelArgs


class S3RFBackbone(Backbone):
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
        s3_vocab_size: int,
        s3_embedding_dim: int,
        output_channels: int,
        reference_dim: int,
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
        input_channels = s3_embedding_dim + output_channels if prev_cond else s3_embedding_dim
        self.input_channels = s3_embedding_dim
        self.s3_embed = nn.Embedding(s3_vocab_size, s3_embedding_dim)
        self.embed = nn.Conv1d(input_channels + output_channels + fourier_dim, dim, kernel_size=7, padding=3)
        self.ref_dim = reference_dim
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

        attn_args = ModelArgs(dim=dim, n_heads=8)
        self.cross_attn = Attention(attn_args)
        self.ref_proj = nn.Linear(self.reference_dim, dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    @staticmethod
    def get_out(out_layer, x):
        x = out_layer(x).transpose(1, 2)
        return x

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, x: torch.Tensor, reference: torch.Tensor,
                bandwidth_id=None, encodec_bandwidth_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.s3_embed(x)
        x = x.transpose(1, 2)

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

        ref = self.ref_proj(reference)
        cond = self.cross_attn(x, ref)
        x = cond + x

        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x + emb_t + emb_b, cond_embedding_id=bandwidth_id)
        x = self.final_layer_norm(x.transpose(1, 2))
        x = self.get_out(self.out, x)
        return x

