from torch import nn
from dataclasses import dataclass
from typing import Any, Optional, Tuple
from rfwave.models import ConvNeXtV2Block

import torch
import torch.nn.functional as F
import math
import numpy as np


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class InputAdaptor(nn.Module):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x, *args):
        return x


@dataclass
class ModelArgs:
    # default hyperparameters for the Llama 7B model
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    hidden_dim: Optional[int] = None
    multiple_of: int = 256  # MLP hidden layer size will be multiple of
    norm_eps: float = 1e-5
    max_seq_len: int = 8192
    dropout: float = 0.0
    qk_norm: bool = True
    return_attn_probs: bool = False


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


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


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


def get_pos_embed(pos_embed_table, start, length, scale=1.):
    # length = length if isinstance(length, int) else length.max()
    scale = scale * torch.ones_like(start, dtype=torch.float32)  # in case scale is a scalar
    pos = start.unsqueeze(1) + (
            torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0) *
            scale.unsqueeze(1)).long()
    # avoid extra long error.
    pos = torch.where(pos < pos_embed_table.size(0), pos, pos_embed_table.size(0) - 1)
    pe = pos_embed_table[pos]
    return pe


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim) if args.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if args.qk_norm else nn.Identity()
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.return_attn_probs = args.return_attn_probs

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
        mask: torch.Tensor
    ):
        bsz, q_seqlen, _ = q_x.shape
        _, kv_seqlen, _ = kv_x.shape

        # QKV
        xq, xk, xv = self.wq(q_x), self.wk(kv_x), self.wv(kv_x)
        xq = xq.view(bsz, q_seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, kv_seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, kv_seqlen, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        # RoPE relative positional embeddings
        xq = apply_rotary_emb(xq, q_freqs_cis)
        xk = apply_rotary_emb(xk, k_freqs_cis)

        # grouped multiquery attention: expand out keys and values
        xk = repeat_kv(xk, self.n_rep)  # (bs, kv_seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, kv_seqlen, n_local_heads, head_dim)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, q_seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # flash implementation
        if self.flash and not self.return_attn_probs:
            output = torch.nn.functional.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)
        else:
            # manual implementation
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, q_seqlen, head_dim)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, q_seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        if self.return_attn_probs:
            return output, scores
        else:
            return output


class SelfAttention(Attention):
    def __init__(self, args: ModelArgs):
        super().__init__(args)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor
    ):
        return super().forward(x, x, freqs_cis, freqs_cis, mask)


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


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cis, mask):
        h = x + self.attention.forward(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class CharInputAdaptor(InputAdaptor):
    def __init__(self, embedding_dim, vocab_size, n_attn_layers=4, n_conv_layers=4,
                 dropout=0., dilation=1):
        super().__init__()
        params = ModelArgs(dim=embedding_dim, vocab_size=vocab_size,
                           n_layers=n_attn_layers, n_heads=8, dropout=dropout)
        self.dim = embedding_dim

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.attn_norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.attn_output = nn.Linear(params.dim, params.dim, bias=False)
        self.convnext = nn.Sequential(
            *[ConvNeXtV2Block(dim=params.dim, intermediate_dim=params.dim * 3, dilation=dilation)
              for _ in range(n_conv_layers)])
        self.output = nn.Linear(params.dim, params.dim)
        self.pad_token = 0

        # some useful precompute for the RoPE relative positional embeddings
        freqs_cis = precompute_freqs_cis(params.dim // params.n_heads, params.max_seq_len)
        self.register_buffer("attn_freqs_cis", freqs_cis, persistent=False)
        freqs_cis = precompute_freqs_cis(params.dim // params.n_heads, params.max_seq_len, theta_rescale_factor=8.)
        self.register_buffer("attn_freqs_cis_eval", freqs_cis, persistent=False)
        freqs_cis = precompute_freqs_cis(params.dim, params.max_seq_len)
        self.register_buffer("conv_freqs_cis", freqs_cis, persistent=False)
        freqs_cis = precompute_freqs_cis(params.dim, params.max_seq_len, theta_rescale_factor=8.)
        self.register_buffer("conv_freqs_cis_eval", freqs_cis, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward_phone(self, tokens: torch.Tensor, phone_start: torch.Tensor):
        _bsz, num_phones = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        freqs_cis = get_pos_embed(self.attn_freqs_cis if self.training else self.attn_freqs_cis_eval,
                                  phone_start, num_phones)
        phone_mask = score_mask_from_bool_mask(tokens == self.pad_token)

        for layer in self.layers:
            h = layer(h, freqs_cis, mask=phone_mask)
        h = self.attn_norm(h)
        h = self.attn_output(h)
        non_padding = (tokens != self.pad_token).float()
        return h * non_padding.unsqueeze(2)

    def expand(self, encoded_phone, lengths):
        out = []
        for phn, l in zip(encoded_phone, lengths):
            out.append(phn.repeat_interleave(l, dim=0))
        return torch.stack(out, dim=0)

    def forward(self, tokens: torch.Tensor, token_frames: torch.Tensor,
                phone_start: torch.Tensor, frame_start: torch.Tensor, *args):
        encoded_phone = self.forward_phone(tokens, phone_start)
        expanded_phone = self.expand(encoded_phone, token_frames)
        non_padding = (expanded_phone.abs().sum(2) > 0.).float()
        num_frames = torch.sum(token_frames, dim=1).long()
        freqs_cis = get_pos_embed(self.conv_freqs_cis if self.training else self.conv_freqs_cis_eval,
                                  frame_start, num_frames.max())
        expanded_phone = apply_rotary_emb(expanded_phone, freqs_cis)
        output = self.convnext(expanded_phone.transpose(1, 2))
        output = self.output(output.transpose(1, 2))
        output = output * non_padding.unsqueeze(2)
        return output.transpose(1, 2)


class CrossAttTransformerBlock(nn.Module):
    def __init__(self, layer_id, args: ModelArgs, modulate=False):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = SelfAttention(args)
        self.cross_attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.layer_id = layer_id
        if modulate:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(self.dim, 9 * self.dim, bias=True))
            self.attention_norm = nn.LayerNorm(args.dim, elementwise_affine=False, eps=args.norm_eps)
            self.cross_attention_norm = nn.LayerNorm(args.dim, elementwise_affine=False, eps=args.norm_eps)
            self.ffn_norm = nn.LayerNorm(args.dim, elementwise_affine=False, eps=args.norm_eps)
        else:
            self.adaLN_modulation = None
            self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
            self.cross_attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
            self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, context, x_freqs_cis, c_freqs_cis, x_mask, c_mask, mod_c=None):
        if self.adaLN_modulation is None:
            h = x + self.attention(self.attention_norm(x), x_freqs_cis, x_mask)
            h = h + self.cross_attention(
                self.cross_attention_norm(h), context, x_freqs_cis, c_freqs_cis, c_mask)
            out = h + self.feed_forward(self.ffn_norm(h))
        else:
            assert mod_c is not None
            (shift_msa, scale_msa, gate_msa, shift_crs, scale_crs, gate_crs,
             shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(mod_c).chunk(9, dim=1)
            h = x + (gate_msa.unsqueeze(1) * self.attention(
                modulate(self.attention_norm(x), shift_msa, scale_msa), x_freqs_cis, x_mask))
            h = h + (gate_crs.unsqueeze(1) * self.cross_attention(
                modulate(self.cross_attention_norm(h), shift_crs, scale_crs),
                context, x_freqs_cis, c_freqs_cis, c_mask))
            out = h + (gate_mlp.unsqueeze(1) * self.feed_forward(
                modulate(self.ffn_norm(h), shift_mlp, scale_mlp)))
        return out


class AlignmentBlock(nn.Module):
    def __init__(self, dim, ctx_dim):
        super().__init__()
        args = ModelArgs(dim, n_heads=1, multiple_of=128, return_attn_probs=True)
        self.dim = dim
        self.ctx_proj = nn.Conv1d(ctx_dim, args.dim, 1) if ctx_dim != args.dim else nn.Identity()
        self.align_attn = Attention(args)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(self.dim, 3 * self.dim, bias=True))
        self.cross_attention_norm = nn.LayerNorm(args.dim, elementwise_affine=False, eps=args.norm_eps)
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
            if pn.endswith('wo.weight'):  # attention output weights
                torch.nn.init.trunc_normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.num_layers))
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, context, x_freqs_cis, c_freqs_cis, x_mask, c_mask, mod_c):
        context = self.ctx_proj(context).transpose(1, 2)
        rpt = self.dim // x_freqs_cis.size(2)
        x_freqs_cis = x_freqs_cis.repeat_interleave(rpt, dim=2)
        c_freqs_cis = c_freqs_cis.repeat_interleave(rpt, dim=2)
        shift_crs, scale_crs, gate_crs = self.adaLN_modulation(mod_c).chunk(3, dim=1)
        h, attn = self.align_attn(
            modulate(self.cross_attention_norm(x), shift_crs, scale_crs),
            context, x_freqs_cis, c_freqs_cis, c_mask)
        h = h + (gate_crs.unsqueeze(1) * h)
        return h, attn


class ContextBlock(nn.Module):
    def __init__(self, args: ModelArgs, ctx_dim, n_attn_layers=4, modulate=False):
        super().__init__()
        self.ctx_proj = nn.Conv1d(ctx_dim, args.dim, 1) if ctx_dim != args.dim else nn.Identity()
        self.ctx_attention = nn.ModuleList([CrossAttTransformerBlock(i, args, modulate=modulate)
                                            for i in range(n_attn_layers)])
        if modulate:
            self.attn_norm = nn.LayerNorm(args.dim, elementwise_affine=False, eps=args.norm_eps)
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(args.dim, 2 * args.dim, bias=True))
        else:
            self.attn_norm = RMSNorm(args.dim, eps=args.norm_eps)
            self.adaLN_modulation = None
        self.attn_output = nn.Linear(args.dim, args.dim, bias=False)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        self.blocks.apply(_basic_init)
        for pn, p in self.blocks.named_parameters():
            if pn.endswith('wo.weight'):  # attention output weights
                torch.nn.init.trunc_normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.num_layers))

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.ctx_attention:
            if block.adaLN_modulation is not None:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        # Zero-out output layers:
        if self.adaLN_modulation is not None:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.trunc_normal_(self.attn_output.weight, mean=0., std=0.02)

    def forward(self, x, context, x_freqs_cis, c_freqs_cis, x_mask, c_mask, mod_c=None):
        c = self.ctx_proj(context)
        c = c.transpose(1, 2)
        h = x

        for layer in self.ctx_attention:
            h = layer(h, c, x_freqs_cis, c_freqs_cis, x_mask, c_mask, mod_c=mod_c)

        if self.adaLN_modulation is None:
            h = self.attn_output(self.attn_norm(h))
        else:
            assert mod_c is not None
            shift, scale = self.adaLN_modulation(mod_c).chunk(2, dim=1)
            h = self.attn_output(modulate(self.attn_norm(h), shift, scale))
        return h


class CtxCharInputAdaptor(InputAdaptor):
    def __init__(self, embedding_dim, vocab_size, ctx_dim,
                 n_attn_layers=4, n_conv_layers=4, n_ctx_layers=4, dropout=0.):
        super().__init__()
        params = ModelArgs(dim=embedding_dim, vocab_size=vocab_size,
                           n_layers=n_attn_layers, n_heads=8, dropout=dropout)
        self.dim = embedding_dim

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.attn_norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.attn_output = nn.Linear(params.dim, params.dim, bias=False)
        self.ctx_proj = nn.Sequential(
            nn.Conv1d(ctx_dim, params.dim, 1),
            *[ConvNeXtV2Block(params.dim, params.dim*3) for _ in range(n_ctx_layers)])
        self.ctx_attn = ContextBlock(
            params, ctx_dim=params.dim, n_attn_layers=n_conv_layers)
        self.pad_token = 0

        # some useful precompute for the RoPE relative positional embeddings
        freqs_cis = precompute_freqs_cis(params.dim // params.n_heads, params.max_seq_len)
        self.register_buffer("attn_freqs_cis", freqs_cis, persistent=False)
        freqs_cis = precompute_freqs_cis(params.dim // params.n_heads, params.max_seq_len, theta_rescale_factor=8.)
        self.register_buffer("attn_freqs_cis_eval", freqs_cis, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward_phone(self, tokens: torch.Tensor, phone_start: torch.Tensor):
        _bsz, num_phones = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        freqs_cis = get_pos_embed(self.attn_freqs_cis if self.training else self.attn_freqs_cis_eval,
                                  phone_start, num_phones)
        phone_mask = score_mask_from_bool_mask(tokens == self.pad_token)

        for layer in self.layers:
            h = layer(h, freqs_cis, mask=phone_mask)

        h = self.attn_norm(h)
        h = self.attn_output(h)
        non_padding = (tokens != self.pad_token).float()
        return h * non_padding.unsqueeze(2)

    def expand(self, encoded_phone, lengths):
        out = []
        for phn, l in zip(encoded_phone, lengths):
            out.append(phn.repeat_interleave(l, dim=0))
        return torch.stack(out, dim=0)

    def forward(self, tokens: torch.Tensor, token_frames: torch.Tensor,
                phone_start: torch.Tensor, frame_start: torch.Tensor,
                context: torch.Tensor, context_lengths: torch.Tensor, *args):
        # context: [b, c, t]
        encoded_phone = self.forward_phone(tokens, phone_start)
        expanded_phone = self.expand(encoded_phone, token_frames)
        non_padding = (expanded_phone.abs().sum(2) > 0.).float()
        num_frames = torch.sum(token_frames, dim=1).long()

        freqs_cis = get_pos_embed(self.attn_freqs_cis if self.training else self.attn_freqs_cis_eval,
                                  frame_start, num_frames.max())
        x_mask = score_mask_from_bool_mask(non_padding == 0)
        ctx_freqs_cis = get_pos_embed(self.attn_freqs_cis if self.training else self.attn_freqs_cis_eval,
                                      torch.zeros_like(frame_start), context.size(2))
        ctx_mask = score_mask(context_lengths)

        context = self.ctx_proj(context)
        output = self.ctx_attn(
            expanded_phone, context, freqs_cis, ctx_freqs_cis, x_mask, ctx_mask)

        output = output * non_padding.unsqueeze(2)
        return output.transpose(1, 2)


class Ctx2CharInputAdaptor(InputAdaptor):
    def __init__(self, embedding_dim, vocab_size, ctx_dim,
                 n_attn_layers=4, n_conv_layers=4, n_ctx_layers=4,
                 dropout=0., drop_ctx=0.5):
        super().__init__()
        params = ModelArgs(dim=embedding_dim, vocab_size=vocab_size,
                           n_layers=n_attn_layers, n_heads=8, dropout=dropout)
        self.dim = embedding_dim
        self.drop_ctx = drop_ctx

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.attn_norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.attn_output = nn.Linear(params.dim, params.dim, bias=False)
        self.ctx_proj = nn.Sequential(
            nn.Conv1d(ctx_dim, params.dim, 1),
            *[ConvNeXtV2Block(params.dim, params.dim*3) for _ in range(n_ctx_layers)])
        self.ctx_attn = ContextBlock(
            params, ctx_dim=params.dim*2, n_attn_layers=n_conv_layers)
        self.pad_token = 0

        # some useful precompute for the RoPE relative positional embeddings
        freqs_cis = precompute_freqs_cis(params.dim // params.n_heads, params.max_seq_len)
        self.register_buffer("attn_freqs_cis", freqs_cis, persistent=False)
        freqs_cis = precompute_freqs_cis(params.dim // params.n_heads, params.max_seq_len, theta_rescale_factor=8.)
        self.register_buffer("attn_freqs_cis_eval", freqs_cis, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward_phone(self, tokens: torch.Tensor, phone_start: torch.Tensor):
        _bsz, num_phones = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        freqs_cis = get_pos_embed(self.attn_freqs_cis if self.training else self.attn_freqs_cis_eval,
                                  phone_start, num_phones)
        phone_mask = score_mask_from_bool_mask(tokens == self.pad_token)

        for layer in self.layers:
            h = layer(h, freqs_cis, mask=phone_mask)

        h = self.attn_norm(h)
        h = self.attn_output(h)
        non_padding = (tokens != self.pad_token).float()
        return h * non_padding.unsqueeze(2)

    def expand(self, encoded_phone, lengths):
        l = torch.max(lengths.sum(1))
        out = torch.zeros([encoded_phone.size(0), l, encoded_phone.size(2)], device=encoded_phone.device)
        for i, (phn, l) in enumerate(zip(encoded_phone, lengths)):
            out[i, :l.sum()] = phn.repeat_interleave(l, dim=0)
        return out

    def forward_ctx(self, context: torch.Tensor, context_lengths: torch.Tensor,
                    ctx_tokens: torch.Tensor, ctx_token_frames: torch.Tensor):
        if self.training:
            if np.random.uniform() < self.drop_ctx:
                drop_ctx, drop_tok = (True, False) if np.random.uniform() < 0.5 else (False, True)
            else:
                drop_ctx, drop_tok = False, False
        else:
            drop_ctx = ctx_tokens is None
            drop_tok = ctx_tokens is None
            assert not (drop_ctx and drop_tok)
        bs, _, l = context.shape
        context = (self.ctx_proj(context) if not drop_ctx else
                   torch.zeros([bs, self.dim, l], device=context.device))
        if not drop_tok:
            encoded_ctx_phone = self.forward_phone(ctx_tokens, _get_start(ctx_tokens, None))
            expanded_ctx_phone = self.expand(encoded_ctx_phone, ctx_token_frames)
            expanded_ctx_phone = expanded_ctx_phone.transpose(1, 2)
        else:
            expanded_ctx_phone = torch.zeros([bs, self.dim, l], device=context.device)
        context = torch.cat([expanded_ctx_phone, context], dim=1)
        return context

    def forward(self, tokens: torch.Tensor, token_frames: torch.Tensor,
                phone_start: torch.Tensor, frame_start: torch.Tensor,
                context: torch.Tensor, context_lengths: torch.Tensor,
                ctx_tokens: torch.Tensor, ctx_token_frames: torch.Tensor):
        assert torch.all(context_lengths == ctx_token_frames.sum(1))
        # context: [b, c, t]
        encoded_phone = self.forward_phone(tokens, phone_start)
        expanded_phone = self.expand(encoded_phone, token_frames)
        non_padding = (expanded_phone.abs().sum(2) > 0.).float()
        num_frames = torch.sum(token_frames, dim=1).long()

        freqs_cis = get_pos_embed(self.attn_freqs_cis if self.training else self.attn_freqs_cis_eval,
                                  frame_start, num_frames.max())
        x_mask = score_mask_from_bool_mask(non_padding == 0)
        ctx_freqs_cis = get_pos_embed(self.attn_freqs_cis if self.training else self.attn_freqs_cis_eval,
                                      torch.zeros_like(frame_start), context.size(2))
        ctx_mask = score_mask(context_lengths)

        context = self.forward_ctx(context, context_lengths, ctx_tokens, ctx_token_frames)
        output = self.ctx_attn(
            expanded_phone, context, freqs_cis, ctx_freqs_cis, x_mask, ctx_mask)

        output = output * non_padding.unsqueeze(2)
        return output.transpose(1, 2)


class DurInputAdaptor(InputAdaptor):
    def __init__(self, embedding_dim, vocab_size, n_attn_layers=4, dropout=0.):
        super().__init__()
        params = ModelArgs(dim=embedding_dim, vocab_size=vocab_size,
                           n_layers=n_attn_layers, n_heads=8, dropout=dropout)
        self.dim = embedding_dim

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.attn_norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.attn_output = nn.Linear(params.dim, params.dim, bias=False)
        self.pad_token = 0
        freqs_cis = precompute_freqs_cis(params.dim // params.n_heads, params.max_seq_len)
        self.register_buffer("attn_freqs_cis", freqs_cis, persistent=False)
        freqs_cis = precompute_freqs_cis(params.dim // params.n_heads, params.max_seq_len, theta_rescale_factor=8.)
        self.register_buffer("attn_freqs_cis_eval", freqs_cis, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor):
        _bsz, num_phones = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        phone_start = torch.zeros([_bsz], dtype=torch.long, device=h.device)

        freqs_cis = get_pos_embed(self.attn_freqs_cis if self.training else self.attn_freqs_cis_eval,
                                  phone_start, num_phones)
        phone_mask = score_mask_from_bool_mask(tokens == self.pad_token)

        for layer in self.layers:
            h = layer(h, freqs_cis, mask=phone_mask)
        h = self.attn_norm(h)
        h = self.attn_output(h)
        non_padding = (tokens != self.pad_token).float()
        out = h * non_padding.unsqueeze(2)
        return out.transpose(1, 2)


class E2ECtxCharInputAdaptor(InputAdaptor):
    def __init__(self, embedding_dim, vocab_size, ctx_dim, num_layers=4):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.ctx_proj = nn.Conv1d(ctx_dim, embedding_dim, kernel_size=1)
        self.tok_blocks = nn.Sequential(*[ConvNeXtV2Block(embedding_dim, embedding_dim*3) for _ in range(num_layers)])
        self.ctx_blocks = nn.Sequential(*[ConvNeXtV2Block(embedding_dim, embedding_dim*3) for _ in range(num_layers)])

    def forward(self, tokens, ctx):
        te = self.tok_embeddings(tokens).transpose(1, 2)
        ce = self.ctx_proj(ctx)
        te = self.tok_blocks(te)
        ce = self.ctx_blocks(ce)
        return torch.cat([te, ce], dim=2)


class InputAdaptorProject(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.linear = nn.Linear(input_channels, output_channels)

    def forward(self, x, pad_val=0.):
        non_padding = (x.abs().sum(1, keepdim=True) > 0.).float()
        out = self.linear(x.transpose(1, 2)).transpose(1, 2)
        return out * non_padding + pad_val * (1. - non_padding)
