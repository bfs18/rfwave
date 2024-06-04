import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from rfwave.modules import GRN, ConvNeXtV2Block


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.):
    # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
    # https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return torch.cat([freqs_cos, freqs_sin], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if ndim == 4:
        assert freqs_cis.shape == (x.shape[0], x.shape[1], x.shape[-1])
        shape = [d if i in (0, 1, ndim - 1) else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(shape)
    elif ndim == 3:
        assert freqs_cis.shape == x.shape
        return freqs_cis
    else:
        raise ValueError(f"Unsupported ndim {ndim}")


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
):
    if freqs_cis is None:
        return x

    x_r, x_i = x.float().reshape(x.shape[:-1] + (-1, 2)).unbind(-1)
    freqs_cos, freqs_sin = torch.chunk(freqs_cis, 2, dim=-1)
    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, x_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, x_r)

    # apply rotation using real numbers
    x_out_r = x_r * freqs_cos - x_i * freqs_sin
    x_out_i = x_r * freqs_sin + x_i * freqs_cos

    # flatten last two dimensions
    x_out = torch.stack([x_out_r, x_out_i], dim=-1).flatten(x_out_r.ndim - 1)

    return x_out.type_as(x)


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


def score_mask_from_bool_mask(bool_mask):
    phone_mask = torch.zeros(*bool_mask.size(), device=bool_mask.device)
    phone_mask.masked_fill_(bool_mask, float('-inf'))
    phone_mask = phone_mask.unsqueeze(1).unsqueeze(2)
    return phone_mask


def _get_start(tensor, start):
    return (torch.zeros([tensor.size(0)], dtype=torch.long, device=tensor.device)
            if start is None else start)


def _get_len(tensor, length):
    return (torch.ones([tensor.size(0)], dtype=torch.long, device=tensor.device) * tensor.size(2)
            if length is None else length)


def get_pos_embed_indices(start, length, max_pos, scale=1.):
    # length = length if isinstance(length, int) else length.max()
    scale = scale * torch.ones_like(start, dtype=torch.float32)  # in case scale is a scalar
    pos = start.unsqueeze(1) + (
            torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0) *
            scale.unsqueeze(1)).long()
    # avoid extra long error.
    pos = torch.where(pos < max_pos, pos, max_pos - 1)
    return pos


class ConvFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, drop: float):
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
        self.dropout = nn.Dropout(drop)

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


class MLP(nn.Module):
    def __init__(
            self,
            dim,
            hidden_dim=None,
            out_dim=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_dim = out_dim or dim
        hidden_dim = hidden_dim or dim * 4
        bias = [bias] * 2
        drop_probs = [drop] * 2

        self.fc1 = nn.Linear(dim, hidden_dim, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_dim) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256, drop: float = 0.):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class CrossAttention(nn.Module):
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

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # use flash attention or a manual implementation?
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        q_freqs_cis: torch.Tensor,
        k_freqs_cis: torch.Tensor,
        mask: torch.Tensor) -> torch.Tensor:
        bsz, q_seqlen, ch = q_x.shape
        _, kv_seqlen, _ = kv_x.shape

        q = self.q(q_x).reshape(bsz, q_seqlen, self.num_heads, self.head_dim)
        kv = self.kv(kv_x).reshape(bsz, kv_seqlen, 2, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rotary_emb(q, q_freqs_cis)
        k = apply_rotary_emb(k, k_freqs_cis)

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
            attn = attn.float().softmax(dim=-1).type_as(attn)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).contiguous().reshape(bsz, q_seqlen, ch)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(CrossAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor
    ):
        return super().forward(x, x, freqs_cis, freqs_cis, mask)


# https://github.com/lucidrains/vector-quantize-pytorch/blob/ce3433256e40328de4a5bf06fadfcda2a227e696/vector_quantize_pytorch/vector_quantize_pytorch.py#L29
# def cdist(x, y):
#     x2 = reduce(x ** 2, 'b n d -> b n', 'sum')
#     y2 = reduce(y ** 2, 'b n d -> b n', 'sum')
#     xy = einsum('b i d, b j d -> b i j', x, y) * -2
#     return (rearrange(x2, 'b i -> b i 1') + rearrange(y2, 'b j -> b 1 j') + xy).clamp(min = 0).sqrt()
def cdist(x, y):
    # x, y: batch_size, num_heads, seq_len, head_dim
    dt = x.dtype
    x = x.float()
    y = y.float()
    x2 = x.pow(2).sum(dim=-1)
    y2 = y.pow(2).sum(dim=-1)
    xy = torch.einsum('b k i d , b k j d -> b k i j', x, y)
    dist = x2.unsqueeze(-1) + y2.unsqueeze(-2) - 2 * xy
    return dist.clamp(0.).to(dt)


class QueryProj(nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.pools = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(nn.Conv1d(dim, dim, kernel_size=3, padding=1))
            self.norms.append(nn.LayerNorm(dim))
            self.pools.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        self.num_layers = num_layers

    def forward(self, x):
        xs = [x]
        for i in range(self.num_layers):
            x = F.silu(self.convs[i](x))
            x = self.norms[i](x.transpose(1, 2)).transpose(1, 2)
            x = self.pools[i](x)
            xs.append(x)

        out = xs[0]
        for i in range(1, self.num_layers):
            xi = F.interpolate(xs[i], scale_factor=2 ** i, mode='nearest')
            out = out + xi[..., :out.size(-1)]
        return out


class QueryDownsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(dim)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = F.silu(self.conv(x))
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.pool(x)
        return x


class CrossAttentionWithPrior(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        num_proj_layers: int = 2,
        type: str = 'gaussian'
    ) -> None:
        assert type in ['gaussian', 'dot_product']
        super().__init__()
        print(f"{num_heads} heads for alignment")
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        # add conv layers as standalone attention.
        self.q_ds = QueryDownsample(dim)
        if num_proj_layers is not None and num_proj_layers > 0:
            self.q_proj = QueryProj(dim, num_proj_layers)
        else:
            self.q_proj = None
        self.score_conv = nn.Conv2d(num_heads, num_heads, kernel_size=(45, 5), padding=(22, 2))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # self.prior_strength = nn.Parameter(torch.ones([self.num_heads]) * 0.1)
        self.prior_strength = 0.1
        self.type = type
        self.temperature = 1.

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        q_freqs_cis: torch.Tensor,
        k_freqs_cis: torch.Tensor,
        mask: torch.Tensor,
        attn_prior: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        bsz, q_seqlen, ch = q_x.shape
        _, kv_seqlen, _ = kv_x.shape

        q_x = self.q_ds(q_x.transpose(1, 2))
        if self.q_proj is not None:
            q_x = self.q_proj(q_x)
        q_x = q_x.transpose(1, 2)

        q = self.q(q_x).reshape(bsz, (q_seqlen + 1) // 2, self.num_heads, self.head_dim)
        k = self.k(kv_x).reshape(bsz, kv_seqlen, self.num_heads, self.head_dim)
        v = self.v(kv_x).reshape(bsz, kv_seqlen, self.num_heads, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)

        # use as absolute positional embedding.
        ps = np.sqrt(self.prior_strength)
        q_freqs_cis = F.pad(q_freqs_cis, (0, 0, 0, 1), mode='replicate')[:, 1::2]
        q = q + q_freqs_cis.unsqueeze(2) * ps
        k = k + k_freqs_cis.unsqueeze(2) * ps

        # (bs, n_local_heads, q_seqlen, head_dim)
        q = q.transpose(1, 2).float()
        k = k.transpose(1, 2).float()
        v = v.transpose(1, 2)

        if self.type == 'gaussian':
            attn = -cdist(q * self.scale, k * self.scale)
        elif self.type == 'dot_product':
            attn = q @ k.transpose(-2, -1) * self.scale
        else:
            raise ValueError(f'Unknown attention type {self.type}')
        temperature = torch.linspace(1, 10, self.num_heads, device=attn.device)
        attn = attn / temperature.view(1, self.num_heads, 1, 1)
        attn = attn + self.score_conv(attn)

        if mask is not None:
            attn = attn + mask
        attn = attn.float().softmax(dim=-1).type_as(attn)
        if attn_prior is not None:
            attn_prior = F.pad(attn_prior, (0, 0, 0, 1), mode='replicate')[:, 1::2]
            attn = torch.exp(torch.log(attn.clamp_min(1e-8)) +
                             torch.log(attn_prior.unsqueeze(1).clamp_min(1e-8)))
            # attn = attn + attn_prior.unsqueeze(1) * self.prior_strength
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
        attn = F.interpolate(attn, size=(attn.size(2) * 2, kv_seqlen), mode='nearest')[:, :, :q_seqlen]
        attn_before_drop = attn
        attn = self.attn_drop(attn)
        x = (attn @ v).type_as(v)

        x = x.transpose(1, 2).contiguous().reshape(bsz, q_seqlen, ch)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_before_drop
