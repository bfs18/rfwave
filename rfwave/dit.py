import torch
import math

from typing import Optional
from torch import nn
from torch.nn import functional as F
from rfwave.models import Backbone, Base2FourierFeatures
from rfwave.input import ModelArgs, ContextBlock, AlignmentBlock
from rfwave.attention import (Attention, FeedForward, ConvFeedForward, MLP, precompute_freqs_cis,  RMSNorm,
                              get_pos_embed_indices, modulate, score_mask, _get_len, _get_start, sequence_mask)
from rfwave.dataset import get_exp_length
from rfwave.standalone_alignment import EmptyAlignmentBlock, gaussian_prior, binarize_attention


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
        # self.feed_forward = MLP(dim=dim, hidden_dim=self.intermediate_dim, drop=dropout,
        #                         act_layer=lambda: nn.GELU(approximate="tanh"))
        self.feed_forward = FeedForward(
            dim=dim, hidden_dim=self.intermediate_dim, drop=dropout, multiple_of=256)
        # self.feed_forward = ConvFeedForward(
        #     dim=dim, hidden_dim=self.intermediate_dim, multiple_of=256, drop=dropout)
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


class ModLayerNorm(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
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
        self.dim = dim
        if self.with_fourier_features:
            self.fourier_module = Base2FourierFeatures(start=6, stop=8, step=1)
            fourier_dim = output_channels * 2 * (
                    (self.fourier_module.stop - self.fourier_module.start) // self.fourier_module.step)
        else:
            fourier_dim = 0
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels + output_channels + fourier_dim, dim, kernel_size=7, padding=3)
        self.embed_mod = ModLayerNorm(dim)
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, intermediate_dim, dropout) for _ in range(num_layers)])
        self.final_mod = ModLayerNorm(dim)
        self.final_out = nn.Linear(dim, output_channels)
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
        self.register_buffer("attn_freqs_cis", precompute_freqs_cis(dim//num_heads, 8192), persistent=False)
        self.register_buffer("attn_freqs_cis_eval",
                             precompute_freqs_cis(dim//num_heads, 8192, theta_rescale_factor=8.), persistent=False)
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
            if (pn.endswith('proj.weight') or pn.endswith('fc2.weight') or
                    pn.endswith('pwconv2.weight') or pn.endswith('w3.weight')):
                torch.nn.init.trunc_normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.num_layers))

        # Initialize input embed:
        nn.init.trunc_normal_(self.embed.weight, std=0.02)
        nn.init.constant_(self.embed_mod.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.embed_mod.adaLN_modulation[-1].bias, 0)

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
        nn.init.constant_(self.final_mod.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_mod.adaLN_modulation[-1].bias, 0)
        nn.init.trunc_normal_(self.final_out.weight, mean=0., std=0.02)

    def get_pos_embed(self, start, length, scale=1., eval_theta_rescale=False):
        # TODO: theta_rescale performs better at evaluation for dit vocoder.
        if eval_theta_rescale:
            attn_freqs_cis = self.attn_freqs_cis if self.training else self.attn_freqs_cis_eval
        else:
            attn_freqs_cis = self.attn_freqs_cis
        pos = get_pos_embed_indices(start, length, max_pos=attn_freqs_cis.size(0), scale=scale)
        return attn_freqs_cis[pos]

    def forward(self, z_t, t, x, bandwidth_id=None, start=None, length=None, encodec_bandwidth_id=None):
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
            be = torch.zeros_like(te)
        if self.encodec_bandwidth_embed is not None:
            assert encodec_bandwidth_id is not None
            ee = self.encodec_bandwidth_embed(encodec_bandwidth_id)
        else:
            ee = torch.zeros_like(te)
        c = be + te + ee

        x = x.transpose(1, 2)
        x = self.embed_mod(x, c)
        start = _get_start(z_t, start)
        length = _get_len(z_t, length)  # length is None
        freq_cis = self.get_pos_embed(start, length.max())
        mask = score_mask(length)
        for block in self.blocks:
            x = block(x, c, freq_cis, mask)
        x = self.final_out(self.final_mod(x, c))
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
        self.rad_align = False
        self.standalone_align = False
        self.standalone_distill = False

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


class RefEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.LayerNorm(dim),
            nn.Linear(dim, dim), nn.SiLU(), nn.LayerNorm(dim),
            nn.Linear(dim, dim))

    def forward(self, ref, ref_length):
        ref_mask = sequence_mask(ref_length)
        ref_out = ref * ref_mask.unsqueeze(1).float()
        avg_out = ref_out.sum(-1) / ref_length.unsqueeze(-1).float()
        out = self.proj(avg_out)
        return out.unsqueeze(-1)


class DiTRFE2ETTSMultiTaskBackbone(Backbone):
    def __init__(
        self,
        input_channels: int,
        output_channels1: int,
        output_channels2: int,
        dim: int,
        intermediate_dim: int,
        num_layers1: int,
        num_layers2: int,
        num_ctx_layers: int,
        num_bands: int,
        num_heads: int = 6,
        dropout: float = 0.,
        pe_scale: float = 1000.,
        with_fourier_features: bool = True,
        rad_align: bool = True,
        standalone_align: bool = False,
        standalone_distill: Optional[bool] = None,
        align_attention_type: Optional[str] = None,
        num_align_ds_layers: Optional[int] = None,
        num_align_proj_layers: Optional[int] = None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels1 = output_channels1
        self.output_channels2 = output_channels2
        self.num_bands = num_bands
        self.rad_align = rad_align
        self.standalone_align = standalone_align
        self.standalone_distill = standalone_distill
        self.dim = dim
        assert not (self.rad_align and self.standalone_align)
        # standalone distill must be assigned when using standalone_align
        assert not standalone_align or (standalone_align and self.standalone_distill is not None)

        params = ModelArgs(dim=dim, n_heads=num_heads, dropout=dropout)
        self.z_t1_proj = nn.Conv1d(output_channels1, dim, 1)
        if self.rad_align or (self.standalone_align and self.standalone_distill):
            assert align_attention_type is not None
            assert num_align_ds_layers is not None and num_align_ds_layers >= 0
            assert num_align_proj_layers is not None and num_align_proj_layers >= 0
            assert num_ctx_layers % 2 == 0 and num_ctx_layers // 2 >= 1
            self.cross_attn = ContextBlock(params, input_channels, num_ctx_layers, modulate=True)
            self.align_block = AlignmentBlock(  # only 1 head for alignment.
                dim, input_channels, 1, num_query_ds_layers=num_align_ds_layers,
                num_proj_layers=num_align_proj_layers, attention_type=align_attention_type)
        elif self.standalone_align and not self.standalone_distill:
            self.cross_attn = ContextBlock(params, input_channels, num_ctx_layers, modulate=True)
            self.sa_align_block = EmptyAlignmentBlock(dim, input_channels)
        else:
            self.cross_attn = ContextBlock(params, input_channels, num_ctx_layers, modulate=True)
        self.ref_embed = RefEmbedding(input_channels)
        print(f"input channels {input_channels} dim {dim}")

        self.module = DiTRFBackbone(
            input_channels=dim,
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

    def get_pos_embed(self, start, length, scale=1.):
        # always use the same positional embedding, since the input tokens and reference are not segment
        attn_freqs_cis = self.module.attn_freqs_cis
        pos = get_pos_embed_indices(start, length, max_pos=attn_freqs_cis.size(0), scale=scale)
        return attn_freqs_cis[pos]

    def get_non_pos_embed(self, bsz, length, device):
        # no positional is applied
        sh = (bsz, length, self.module.attn_freqs_cis.size(-1) // 2)
        freqs_cos = torch.ones(sh, device=device)
        freqs_sin = torch.zeros(sh, device=device)
        freq_cis = torch.cat([freqs_cos, freqs_sin], dim=-1)
        return freq_cis

    def time_embed(self, t):
        return self.module.time_embed(t)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, x: torch.Tensor,
                bandwidth_id: torch.Tensor=None, num_tokens=None, ctx_length=None,
                token_exp_scale=None, standalone_attn=None, duration=None):
        # pos_embed for z_t1
        assert num_tokens is not None
        assert ctx_length is not None
        assert token_exp_scale is not None

        z_t1, z_t2 = torch.split(z_t, [self.output_channels1, self.output_channels2], dim=1)
        z_t1 = self.z_t1_proj(z_t1)

        start = _get_start(z_t, None)  # always start from 0 for sentence training.
        # length = torch.round(num_tokens * token_exp_scale).long()
        length = get_exp_length(num_tokens, token_exp_scale)
        z_mask = score_mask(length)
        z_freq_cis = self.get_pos_embed(start, length.max())

        # pos_embed for token and ref
        token_mask = score_mask(num_tokens)
        ref_mask = score_mask(ctx_length)
        zero_start = _get_start(z_t, None)
        token_freq_cis = self.get_pos_embed(zero_start, num_tokens.max(), scale=token_exp_scale)
        # ref_freq_cis = self.get_pos_embed(zero_start, ctx_length.max())
        ref_freq_cis = self.get_non_pos_embed(z_t.size(0), ctx_length.max(), z_t.device)
        ctx_mask = torch.cat([token_mask, ref_mask], dim=-1)
        ctx_freq_cis = torch.cat([token_freq_cis, ref_freq_cis], dim=1)

        x_token, x_ref = torch.split(x, [num_tokens.max(), ctx_length.max()], dim=-1)

        ref_emb = self.ref_embed(x_ref, ctx_length)
        te = self.time_embed(t)
        z_t1 = z_t1.transpose(1, 2)
        if self.rad_align or (self.standalone_align and self.standalone_distill):
            ctx = self.cross_attn(z_t1, x, z_freq_cis, ctx_freq_cis, z_mask, ctx_mask, mod_c=te)
            attn_prior = gaussian_prior(num_tokens, token_exp_scale)
            ctx, attn = self.align_block(ctx, x_token, z_freq_cis, token_freq_cis, token_mask,
                                         mod_c=te, attn_prior=attn_prior)
            rand_attn = attn[torch.arange(attn.size(0)), torch.randint(0, attn.size(1), size=(attn.size(0),))]
            align_ctx = (rand_attn  @ (x_token.detach() + ref_emb).transpose(1, 2)).transpose(-1, -2)
            attn = rand_attn.unsqueeze(1)
        elif self.standalone_align and not self.standalone_distill:
            assert not (standalone_attn is None and duration is None)
            attn = standalone_attn
            ctx = self.cross_attn(z_t1, x, z_freq_cis, ctx_freq_cis, z_mask, ctx_mask, mod_c=te)
            if attn is not None:
                bin_attn = binarize_attention(attn, num_tokens, length)
                align_ctx = (attn.squeeze(1)  @ (x_token.detach() + ref_emb).transpose(1, 2)).transpose(-1, -2)
            else:
                bin_attn = align_ctx = None
            ctx = self.sa_align_block(ctx, x_token, bin_attn, duration, mod_c=te)
        else:
            ctx = self.cross_attn(z_t1, x, z_freq_cis, ctx_freq_cis, z_mask, ctx_mask, mod_c=te)
            align_ctx = ctx.transpose(-1, -2) + ref_emb  # not really
            attn = None
        ctx = ctx.transpose(1, 2)

        return self.module(z_t, t, ctx, bandwidth_id, start=start), attn, align_ctx
