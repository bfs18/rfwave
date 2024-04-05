import torch
import math

from typing import Optional
from torch import nn
from torch.nn import functional as F
from rfwave.models import Backbone, Base2FourierFeatures
from rfwave.input import ModelArgs, ContextBlock, AlignmentBlock
from rfwave.attention import (Attention, FeedForward, ConvFeedForward, precompute_freqs_cis,  RMSNorm,
                              get_pos_embed, modulate, score_mask, _get_len, _get_start)


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
        # self.feed_forward = ConvFeedForward(dim=dim, hidden_dim=self.intermediate_dim,
        #                                     multiple_of=256, dropout=dropout)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=self.intermediate_dim, drop=dropout,
                                        act_layer=lambda: nn.GELU(approximate="tanh"))
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
        self.register_buffer("pos_embed", precompute_freqs_cis(dim//num_heads, 8192), persistent=False)
        self.register_buffer("pos_embed_eval",
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
            if pn.endswith('proj.weight') or pn.endswith('fc2.weight') or pn.endswith('pwconv2.weight'):
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

    def forward(self, z_t, t, x, bandwidth_id=None, start=None, encodec_bandwidth_id=None):
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
        length = _get_len(z_t, None)  # length is None
        freq_cis = get_pos_embed(self.pos_embed if self.training else self.pos_embed_eval, start, length.max())
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
                bandwidth_id: torch.Tensor=None, start=None):
        return self.module(z_t, t, x, bandwidth_id, start=start)


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
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels1 = output_channels1
        self.output_channels2 = output_channels2
        self.num_bands = num_bands

        params = ModelArgs(dim=dim, n_heads=num_heads, dropout=dropout)
        self.z_t1_proj = nn.Conv1d(output_channels1, dim, 1)
        self.cross_attn = ContextBlock(params, input_channels, num_ctx_layers, modulate=True)
        self.align_block = AlignmentBlock(dim, input_channels)

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

    @property
    def pos_embed(self):
        # return self.module.pos_embed if self.training else self.module.pos_embed_eval
        # always use the same positional embedding, since the input tokens and reference are not segment
        return self.module.pos_embed

    def time_embed(self, t):
        return self.module.time_embed(t)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, x: torch.Tensor,
                bandwidth_id: torch.Tensor=None, start=None, token_ref_length=None):
        z_t1, z_t2 = torch.split(z_t, [self.output_channels1, self.output_channels2], dim=1)
        z_t1 = self.z_t1_proj(z_t1)

        # pos_embed for z_t1
        assert start is not None
        assert token_ref_length is not None

        start = _get_start(z_t, start)
        length = _get_len(z_t, None)  # length is None
        z_freq_cis = get_pos_embed(self.pos_embed, start, length.max())
        # pos_embed for token and ref
        token_length, ref_length, token_exp_scale = token_ref_length.unbind(1)
        token_length, ref_length = token_length.long(), ref_length.long()
        token_mask = score_mask(token_length)
        ref_mask = score_mask(ref_length)
        zero_start = _get_start(z_t, None)
        token_freq_cis = get_pos_embed(self.pos_embed, zero_start, token_length.max(), scale=token_exp_scale)
        ref_freq_cis = get_pos_embed(self.pos_embed, zero_start, ref_length.max())
        ctx_mask = torch.cat([token_mask, ref_mask], dim=-1)
        ctx_freq_cis = torch.cat([token_freq_cis, ref_freq_cis], dim=1)

        te = self.time_embed(t)
        z_t1 = z_t1.transpose(1, 2)
        ctx = self.cross_attn(z_t1, x, z_freq_cis, ctx_freq_cis, None, ctx_mask, mod_c=te)
        # before or after cross attention
        ctx, attn = self.align_block(ctx, x, z_freq_cis, ctx_freq_cis, None, ctx_mask, mod_c=te)
        attn, _ = torch.split(attn, [token_length.max(), ref_length.max()], dim=-1)
        attn = attn / attn.sum(dim=-1, keepdim=True)  # renorm attn
        ctx = ctx.transpose(1, 2)
        return self.module(z_t, t, ctx, bandwidth_id, start=start), attn


def find_segment_tokens_(attn, thres=2., win=4):
    expected_frames = attn.sum(0)
    expected_frames = F.pad(expected_frames, (win // 2, win // 2))
    l = expected_frames.size(0)

    for i in range(win // 2, l - win // 2):
        if (torch.all(expected_frames[i - win // 2: i] < thres) and
                torch.all(expected_frames[i: i + win // 2] >= thres)):
            break
    s = i - 2
    for i in reversed(range(win // 2, l - win // 2)):
        if (torch.all(expected_frames[i - win // 2: i] >= thres) and
                torch.all(expected_frames[i: i + win // 2] < thres)):
            break
    e = i - 2
    return s, e


def find_segment_tokens(attn, thres=2., win=4):
    expected_frames = attn.sum(0)
    expected_frames = F.pad(expected_frames, (win // 2, win // 2 - 1))
    used_tokens = expected_frames >= thres
    used_tokens = used_tokens.unfold(0, 4, 1)
    start_patt = torch.tensor([0] * (win // 2) + [1] * (win // 2), dtype=torch.bool, device=attn.device)
    end_patt = torch.tensor([1] * (win // 2) + [0] * (win // 2), dtype=torch.bool, device=attn.device)
    s = torch.where(torch.all(used_tokens == start_patt, dim=1))[0]
    e = torch.where(torch.all(used_tokens == end_patt, dim=1))[0]
    if s.numel() > 0 and e.numel() > 0:
        return s[0], e[-1]
    else:
        return 0, 0


def compute_alignment_loss(attn, start, token_ref_length):
    attn = attn.reshape(-1, *attn.shape[-2:])
    segment_attn = []
    segment_length = []
    aco_length = attn.size(1)
    token_length, _, token_exp_scale = token_ref_length.unbind(1)
    token_length = token_length.long()
    for attn_i, aco_start_i, token_length_i, token_exp_scale_i in zip(
            attn, start, token_length, token_exp_scale):
        attn_i = attn_i[:, :token_length_i]
        s, e = find_segment_tokens(attn_i)
        min_start = aco_start_i / token_exp_scale_i - 10
        max_end = (aco_start_i + aco_length) / token_exp_scale_i + 10
        if min_start <= s < e <= max_end:
            attn_i = attn_i[:, s: e]
            segment_attn.append(attn_i)
            segment_length.append(e - s)
    if len(segment_attn) == 0:
        return 0.
    max_seg_len = max(segment_length)
    batch_size = len(segment_attn)
    attn = torch.zeros([batch_size, aco_length, max_seg_len], device=attn.device)
    target = torch.zeros([batch_size, max_seg_len], device=attn.device, dtype=torch.long)
    for i, (seg_len, seg_attn) in enumerate(zip(segment_length, segment_attn)):
        attn[i, :, :seg_len] = seg_attn
        target[i, :seg_len] = torch.arange(1, seg_len + 1, device=attn.device)
    attn = F.pad(attn, (1, 0, 0, 0, 0, 0), value=0.2)  # prob for blank.
    attn = attn / attn.sum(dim=-1, keepdim=True)
    log_prob = torch.log(attn.clamp_min_(1e-5))
    input_lengths = torch.ones([batch_size], device=attn.device, dtype=torch.long) * aco_length
    target_lengths = torch.tensor(segment_length, device=attn.device, dtype=torch.long)
    loss = F.ctc_loss(log_prob.transpose(1, 0), targets=target, zero_infinity=True,
                      input_lengths=input_lengths, target_lengths=target_lengths, blank=0)
    return loss
