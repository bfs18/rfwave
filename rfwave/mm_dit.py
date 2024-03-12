import collections
import torch
import torch.nn.functional as F
import math
import numpy as np

from itertools import repeat
from functools import partial
from torch import nn


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)


def get_1d_sincos_pos_embed(embed_dim, max_sqe_len):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    pos = np.arange(max_sqe_len)
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb.astype(np.float32)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def score_mask(length, max_length=None):
    seq_mask = sequence_mask(length, max_length)
    sco_mask = torch.zeros_like(seq_mask, dtype=torch.float)
    sco_mask.masked_fill_(~seq_mask, float('-inf'))
    return sco_mask.unsqueeze(1).unsqueeze(2)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class MMAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = RMSNorm,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.m1_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.m1_q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.m1_k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.m2_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.m2_q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.m2_k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.m1_proj = nn.Linear(dim, dim)
        self.m2_proj = nn.Linear(dim, dim)
        self.m1_proj_drop = nn.Dropout(proj_drop)
        self.m2_proj_drop = nn.Dropout(proj_drop)

        # use flash attention or a manual implementation?
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self, x1, x2, mask1=None, mask2=None):
        B1, N1, C1 = x1.shape
        B2, N2, C2 = x2.shape
        assert B1 == B2 and C1 == C2

        m1_qkv = self.m1_qkv(x1).reshape(B1, N1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        m1_q, m1_k, m1_v = m1_qkv.unbind(0)
        m1_q, m1_k = self.m1_q_norm(m1_q), self.m1_k_norm(m1_k)

        m2_qkv = self.m2_qkv(x2).reshape(B2, N2, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        m2_q, m2_k, m2_v = m2_qkv.unbind(0)
        m2_q, m2_k = self.m2_q_norm(m2_q), self.m2_k_norm(m2_k)

        q = torch.cat([m1_q, m2_q], dim=2)
        k = torch.cat([m1_k, m2_k], dim=2)
        v = torch.cat([m1_v, m2_v], dim=2)
        mask = torch.cat([mask1, mask2], dim=-1) if (mask1 is not None and mask2 is not None) else None

        if self.flash:
            x = F.scaled_dot_product_attention(
                q, k, v, mask, dropout_p=self.attn_drop.p if self.training else 0.)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn + mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B1, N1 + N2, C1)
        m1_x, m2_x = x.split([N1, N2], dim=1)
        m1_x = self.m1_proj(m1_x)
        m2_x = self.m2_proj(m2_x)
        m1_x = self.m1_proj_drop(m1_x)
        m2_x = self.m2_proj_drop(m2_x)
        return m1_x, m2_x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MMDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4., **block_args):
        super().__init__()
        self.m1_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.m2_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MMAttention(hidden_size, num_heads=num_heads, qkv_bias=False, qk_norm=True,
                                norm_layer=RMSNorm, **block_args)
        self.m1_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.m2_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.m1_mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.m2_mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.m1_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.m2_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x1, x2, c, x1_mask=None, x2_mask=None):
        (m1_shift_msa, m1_scale_msa, m1_gate_msa, m1_shift_mlp, m1_scale_mlp, m1_gate_mlp
         ) = self.m1_adaLN_modulation(c).chunk(6, dim=1)
        (m2_shift_msa, m2_scale_msa, m2_gate_msa, m2_shift_mlp, m2_scale_mlp, m2_gate_mlp
         ) = self.m2_adaLN_modulation(c).chunk(6, dim=1)
        x1 = modulate(self.m1_norm1(x1), m1_shift_msa, m1_scale_msa)
        x2 = modulate(self.m2_norm1(x2), m2_shift_msa, m2_scale_msa)
        x1, x2 = self.attn(x1, x2, x1_mask, x2_mask)
        x1 = x1 + m1_gate_msa.unsqueeze(1) * x1
        x2 = x2 + m2_gate_msa.unsqueeze(1) * x2
        x1 = x1 + m1_gate_mlp.unsqueeze(1) * self.m1_mlp(modulate(self.m1_norm2(x1), m1_shift_mlp, m1_scale_mlp))
        x2 = x2 + m2_gate_mlp.unsqueeze(1) * self.m2_mlp(modulate(self.m2_norm2(x2), m2_shift_mlp, m2_scale_mlp))
        return x1, x2


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, pe_scale=1000):
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


class MMDiT(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_size=768,
        depth=8,
        num_heads=16,
        mlp_ratio=4.0,
        max_seq_len=4096,
        num_bands=8,
        pe_scale=1000,
    ):
        super().__init__()

        self.t_embed = TimestepEmbedder(hidden_size, pe_scale=pe_scale)
        pos_embed = torch.from_numpy(get_1d_sincos_pos_embed(hidden_size, max_seq_len))
        self.register_buffer("pos_embed", pos_embed, persistent=False)
        self.band_embed = nn.Embedding(num_bands, hidden_size)

        self.m1_proj = nn.Linear(in_channels, hidden_size)
        self.m2_proj = nn.Linear(out_channels, hidden_size)
        self.blocks = nn.ModuleList([
            MMDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)

        # Initialize band embedding:
        nn.init.normal_(self.band_embed.weight, mean=0.0, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.m1_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.m1_adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.m2_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.m2_adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def get_pos_embed(self, start, length):
        pos = start.unsqueeze(1) + torch.arange(length.max(), device=start.device).unsqueeze(0)
        pe = self.pos_embed[pos]
        return pe

    @staticmethod
    def _get_start(tensor, start):
        return (torch.zeros([tensor.size(0)], dtype=torch.long, device=tensor.device)
                if start is None else start)

    @staticmethod
    def _get_len(tensor, length):
        return (torch.ones([tensor.size(0)], dtype=torch.long, device=tensor.device) * tensor.size(1)
                if length is None else length)

    def forward(self,
                x1, x2, t, bandwidth_id, ctx=None,
                x2_start=None, x2_len=None, x1_start=None, x1_len=None, ctx_start=None, ctx_len=None):

        x1_start = self._get_start(x1, x1_start)
        x1_len = self._get_len(x1, x1_len)
        x1_sco_mask = score_mask(x1_len)
        x1_pe = self.get_pos_embed(x1_start, x1_len)
        x1 = self.m1_proj(x1) + x1_pe

        if ctx is not None:
            ctx_start = self._get_start(ctx, ctx_start)
            ctx_len = self._get_len(ctx, ctx_len)
            ctx_sco_mask = score_mask(ctx_len)
            ctx_pe = self.get_pos_embed(ctx_start, ctx_len)
            ctx = self.m1_proj(ctx) + ctx_pe
            x1 = torch.cat([ctx, x1], dim=1)
            x1_sco_mask = torch.cat([ctx_sco_mask, x1_sco_mask], dim=-1)

        x2_start = self._get_start(x2, x2_start)
        x2_len = self._get_len(x2, x2_len)
        x2_sco_mask = score_mask(x2_len)
        x2_pe = self.get_pos_embed(x2_start, x2_len)
        x2 = self.m2_proj(x2) + x2_pe

        te = self.t_embed(t)
        be = self.band_embed(bandwidth_id)
        c = te + be

        for block in self.blocks:
            x1, x2 = block(x1, x2, c, x1_sco_mask, x2_sco_mask)

        x2 = self.final_layer(x2, c)
        return x2


if __name__ == '__main__':
    B = 2
    N1 = 8
    N2 = 7
    C = 256
    mask1 = torch.zeros([B, N1])
    mask1[:, -3:] = float('-inf')
    mask2 = torch.zeros([B, N2])
    mask2[:, -1:] = float('-inf')
    block = MMDiTBlock(256, 8, 4)
    x1 = torch.randn([B, N1, C])
    x2 = torch.randn([B, N2, C])
    c = torch.randn([B, C])
    x1_out, x2_out = block(x1, x2, c, mask1.view(B, 1, 1, N1), mask2.view(B, 1, 1, N2))
    print(x1_out.shape, x2_out.shape, torch.all(x1_out.isfinite()), torch.all(x2_out.isfinite()))

    x1_start = torch.tensor([9, 10], dtype=torch.long)
    x1_len = torch.tensor([8, 7], dtype=torch.long)
    x2_start = torch.tensor([2, 3])
    x2_len = torch.tensor([4, 7])
    ctx_seq = torch.randn([B, 10, C])
    bandwidth_id = torch.tensor([0, 1], dtype=torch.long)
    t = torch.tensor([0.1, 0.2], dtype=torch.float)
    model = MMDiT(C, C)
    out = model(x1=x1, x2=x2, t=t, bandwidth_id=bandwidth_id, ctx=ctx_seq,
                x1_start=x1_start, x1_len=x1_len, x2_start=x2_start, x2_len=x2_len)
    print(out.shape, torch.all(out.isfinite()))
