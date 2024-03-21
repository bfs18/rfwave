import torch
import math

from typing import Optional
from torch import nn
from torch.nn import functional as F
from rfwave.modules import GRN
from rfwave.models import Backbone, Base2FourierFeatures
from rfwave.input import (precompute_freqs_cis,  RMSNorm, apply_rotary_emb, get_pos_embed,
                          _get_len, _get_start, sequence_mask, score_mask)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ConvFF(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, hidden_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(hidden_dim)
        self.pwconv2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # use flash attention or a manual implementation?
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self, x: torch.Tensor, freq_cis: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # (3, bs, q_seqlen, n_local_heads, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rotary_emb(q, freq_cis)
        k = apply_rotary_emb(k, freq_cis)

        # (bs, n_local_heads, q_seqlen, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.flash:
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.attn_drop.p if self.training else 0.)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn + mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).contiguous().reshape(B, N, C)
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, intermediate_dim, dropout=0.):
        super().__init__()
        self.n_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.intermediate_dim = intermediate_dim
        self.norm1 = nn.LayerNorm(self.dim, elementwise_affine=False, eps=1e-6)
        self.attention = Attention(dim=dim, num_heads=num_heads, qkv_bias=False, qk_norm=True,
                                   norm_layer=RMSNorm, attn_drop=dropout, proj_drop=dropout)
        self.norm2 = nn.LayerNorm(self.dim, elementwise_affine=False, eps=1e-6)
        # self.feed_forward = ConvFF(dim=dim, hidden_dim=self.intermediate_dim,
        #                            multiple_of=256, dropout=dropout)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=self.intermediate_dim,
                                        multiple_of=256, dropout=dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(self.dim, 6 * self.dim, bias=True))

    def forward(self, x, c, freqs_cis, mask):
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
         ) = self.adaLN_modulation(c).chunk(6, dim=1)
        h = x + (gate_msa.unsqueeze(1) *
                 self.attention(modulate(self.norm1(x), shift_msa, scale_msa), freqs_cis, mask))
        out = h + (gate_mlp.unsqueeze(1) *
                   self.feed_forward(modulate(self.norm2(h), shift_mlp, scale_mlp)))
        return out


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, pe_scale=1000.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.pe_scale = pe_scale

    def timestep_embedding(self, t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = self.pe_scale * t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTRFBackbone(Backbone):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        num_bands: Optional[int],
        encodec_num_embeddings: Optional[int] = None,
        num_heads: int = 6,
        dropout: float = 0.,
        pe_scale: float = 1000.,
        with_fourier_features: bool = True
    ):
        super().__init__()
        self.prev_cond = False
        self.output_channels = output_channels
        self.with_fourier_features = with_fourier_features
        self.num_bands = num_bands
        self.num_layers = num_layers
        if self.with_fourier_features:
            self.fourier_module = Base2FourierFeatures(start=6, stop=8, step=1)
            fourier_dim = output_channels * 2 * (
                    (self.fourier_module.stop - self.fourier_module.start) // self.fourier_module.step)
        else:
            fourier_dim = 0
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels + output_channels + fourier_dim, dim, kernel_size=7, padding=3)
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, intermediate_dim, dropout) for _ in range(num_layers)])
        self.final = FinalLayer(dim, output_channels)
        self.time_embed = TimestepEmbedder(dim, pe_scale=pe_scale)
        if self.num_bands is not None and self.num_bands > 0:
            self.band_embed = nn.Sequential(nn.Embedding(num_bands, dim), nn.Linear(dim, dim))
        else:
            self.band_embed = None
        if encodec_num_embeddings is not None and encodec_num_embeddings > 0:
            self.encodec_bandwidth_embed = nn.Sequential(
                nn.Embedding(encodec_num_embeddings, dim), nn.Linear(dim, dim))
        else:
            self.encodec_bandwidth_embed = None
        self.register_buffer("pos_embed", precompute_freqs_cis(dim//num_heads, 4096))
        self.register_buffer("pos_embed_eval",
                             precompute_freqs_cis(dim//num_heads, 4096, theta_rescale_factor=8.))
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        self.blocks.apply(_basic_init)
        for pn, p in self.blocks.named_parameters():
            if pn.endswith('proj.weight'):
                torch.nn.init.trunc_normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.num_layers))

        # Initialize input embed:
        nn.init.trunc_normal_(self.embed.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.trunc_normal_(self.time_embed.mlp[0].weight, mean=0., std=0.02)
        nn.init.trunc_normal_(self.time_embed.mlp[2].weight, mean=0., std=0.02)

        # Initialize band embedding:
        if self.band_embed is not None:
            nn.init.trunc_normal_(self.band_embed[0].weight, mean=0., std=0.02)
            nn.init.trunc_normal_(self.band_embed[1].weight, mean=0., std=0.02)
        if self.encodec_bandwidth_embed is not None:
            nn.init.trunc_normal_(self.encodec_bandwidth_embed[0].weight, mean=0., std=0.02)
            nn.init.trunc_normal_(self.encodec_bandwidth_embed[1].weight, mean=0., std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final.adaLN_modulation[-1].bias, 0)
        nn.init.trunc_normal_(self.final.linear.weight, mean=0., std=0.02)

    def forward(self, z_t, t, x, bandwidth_id=None,
                start=None, length=None, encodec_bandwidth_id=None):
        if self.with_fourier_features:
            z_t_f = self.fourier_module(z_t)
            x = self.embed(torch.cat([z_t, x, z_t_f], dim=1))
        else:
            x = self.embed(torch.cat([z_t, x], dim=1))

        te = self.time_embed(t)
        if self.band_embed is not None:
            assert bandwidth_id is not None
            be = self.band_embed(bandwidth_id)
        else:
            be = 0.
        if self.encodec_bandwidth_embed is not None:
            assert encodec_bandwidth_id is not None
            ee = self.encodec_bandwidth_embed(encodec_bandwidth_id)
        else:
            ee = 0.
        c = te + be + ee

        x = x.transpose(1, 2)
        start = _get_start(z_t, start)
        length = _get_len(z_t, None)
        freq_cis = get_pos_embed(self.pos_embed if self.training else self.pos_embed_eval, start, length)
        for block in self.blocks:
            x = block(x, c, freq_cis, None)
        x = self.final(x, c)
        return x.transpose(1, 2)


class DiTRFTTSMultiTaskBackbone(Backbone):
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

        self.module = DiTRFBackbone(
            input_channels=input_channels,
            output_channels=output_channels1 + output_channels2,
            dim=dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers1 + num_layers2,
            num_bands=num_bands,
            encodec_num_embeddings=None,
            num_heads=num_heads,
            dropout=dropout,
            pe_scale=pe_scale,
            with_fourier_features=with_fourier_features)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, x: torch.Tensor,
                bandwidth_id: torch.Tensor=None, start=None, length=None):
        return self.module(z_t, t, x, bandwidth_id, start=start, length=length)
