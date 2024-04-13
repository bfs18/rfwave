import torch

from rfwave.dit import DiTRFBackbone
from rfwave.models import VocosRFBackbone, Backbone


class MixRFTTSMultiTaskBackbone(Backbone):
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
        num_heads: int = 6,
        dropout: float = 0.,
        pe_scale: float = 1000.,
        with_fourier_features: bool = True,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels1 = output_channels1
        self.output_channels2 = output_channels2
        self.num_bands = num_bands

        self.dit_module = DiTRFBackbone(
            input_channels=input_channels,
            output_channels=output_channels1 + output_channels2,
            dim=dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers1,
            num_bands=num_bands,
            encodec_num_embeddings=None,
            num_heads=num_heads,
            dropout=dropout,
            pe_scale=pe_scale,
            with_fourier_features=with_fourier_features)
        self.conv_module = VocosRFBackbone(
            input_channels=input_channels,
            output_channels=output_channels1 + output_channels2,
            dim=dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers2,
            num_bands=num_bands,
            encodec_num_embeddings=None,
            prev_cond=False,
            pe_scale=pe_scale,
            with_fourier_features=False)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, x: torch.Tensor,
                bandwidth_id: torch.Tensor=None, start=None):
        dit_out = self.dit_module(z_t, t, x, bandwidth_id, start=start)
        conv_out = self.conv_module(dit_out, t, x, bandwidth_id, start=start)
        return conv_out
