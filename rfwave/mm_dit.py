import collections
import torch
import torch.nn.functional as F

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


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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
        self.m1_norm1 = RMSNorm(hidden_size)
        self.m2_norm1 = RMSNorm(hidden_size)
        self.attn = MMAttention(hidden_size, num_heads=num_heads, **block_args)
        self.m1_norm2 = RMSNorm(hidden_size)
        self.m2_norm2 = RMSNorm(hidden_size)
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
