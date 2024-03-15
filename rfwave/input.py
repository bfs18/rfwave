from torch import nn
from dataclasses import dataclass
from typing import Any, Optional, Tuple
from rfwave.models import ConvNeXtV2Block

import torch
import torch.nn.functional as F
import math


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
    max_seq_len: int = 2048
    dropout: float = 0.0


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
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
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

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
        if self.flash:
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


class CharInputTransformerAdaptor(InputAdaptor):
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
        freqs_cis = precompute_freqs_cis(params.dim, params.max_seq_len)
        self.register_buffer("conv_freqs_cis", freqs_cis, persistent=False)

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
        freqs_ids = phone_start.unsqueeze(1) + torch.arange(num_phones, device=tokens.device).unsqueeze(0)
        freqs_cis = self.attn_freqs_cis[freqs_ids]

        phone_mask = torch.zeros(*tokens.size(), device=tokens.device)
        phone_mask.masked_fill_(tokens == self.pad_token, float('-inf'))
        phone_mask = phone_mask.unsqueeze(1).unsqueeze(2)

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

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor,
                phone_start: torch.Tensor, frame_start: torch.Tensor):
        encoded_phone = self.forward_phone(tokens, phone_start)
        expanded_phone = self.expand(encoded_phone, lengths)
        non_padding = (expanded_phone.abs().sum(2) > 0.).float()
        num_frames = torch.sum(lengths, dim=1).long()
        freqs_ids = frame_start.unsqueeze(1) + torch.arange(num_frames[0], device=tokens.device).unsqueeze(0)
        freqs_cis = self.conv_freqs_cis[freqs_ids]
        expanded_phone = apply_rotary_emb(expanded_phone, freqs_cis)
        output = self.convnext(expanded_phone.transpose(1, 2))
        output = self.output(output.transpose(1, 2))
        output = output * non_padding.unsqueeze(2)
        return output.transpose(1, 2)


class CrossAttTransformerBlock(nn.Module):
    def __init__(self, layer_id, args: ModelArgs):
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
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.cross_attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, context, x_freqs_cis, c_freqs_cis, x_mask, c_mask):
        h = x + self.attention.forward(self.attention_norm(x), x_freqs_cis, x_mask)
        h = h + self.cross_attention.forward(
            self.cross_attention_norm(h), context, x_freqs_cis, c_freqs_cis, c_mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class ContextBlock(nn.Module):
    def __init__(self, args: ModelArgs, ctx_dim, n_ctx_layers=4, n_attn_layers=4):
        super().__init__()
        self.ctx_convnext = nn.Sequential(
            nn.Conv1d(ctx_dim, args.dim, 1),
            *[ConvNeXtV2Block(dim=args.dim, intermediate_dim=args.dim * 3)
              for _ in range(n_ctx_layers)])
        self.ctx_attention = nn.ModuleList([CrossAttTransformerBlock(i, args) for i in range(n_attn_layers)])
        self.attn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.attn_output = nn.Linear(args.dim, args.dim, bias=False)

    def forward(self, x, context, x_freqs_cis, c_freqs_cis, x_mask, c_mask):
        c = self.ctx_convnext(context)
        c = c.transpose(1, 2)
        h = x
        for layer in self.ctx_attention:
            h = layer(h, c, x_freqs_cis, c_freqs_cis, x_mask, c_mask)
        h = self.attn_norm(h)
        h = self.attn_output(h)
        return h


class CtxCharInputTransformerAdaptor(InputAdaptor):
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
        self.ctx_block = ContextBlock(
            params, ctx_dim=ctx_dim, n_ctx_layers=n_ctx_layers, n_attn_layers=n_conv_layers)
        self.pad_token = 0

        # some useful precompute for the RoPE relative positional embeddings
        freqs_cis = precompute_freqs_cis(params.dim // params.n_heads, params.max_seq_len)
        self.register_buffer("attn_freqs_cis", freqs_cis, persistent=False)

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
        freqs_ids = phone_start.unsqueeze(1) + torch.arange(num_phones, device=tokens.device).unsqueeze(0)
        freqs_cis = self.attn_freqs_cis[freqs_ids]

        phone_mask = torch.zeros(*tokens.size(), device=tokens.device)
        phone_mask.masked_fill_(tokens == self.pad_token, float('-inf'))
        phone_mask = phone_mask.unsqueeze(1).unsqueeze(2)

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

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor,
                phone_start: torch.Tensor, frame_start: torch.Tensor,
                context: torch.Tensor, context_lengths: torch.Tensor):
        # context: [b, c, t]
        encoded_phone = self.forward_phone(tokens, phone_start)
        expanded_phone = self.expand(encoded_phone, lengths)
        non_padding = (expanded_phone.abs().sum(2) > 0.).float()
        num_frames = torch.sum(lengths, dim=1).long()

        freqs_ids = frame_start.unsqueeze(1) + torch.arange(num_frames[0], device=tokens.device).unsqueeze(0)
        freqs_cis = self.attn_freqs_cis[freqs_ids]
        x_mask = torch.zeros_like(non_padding)
        x_mask.masked_fill_(non_padding == 0, float('-inf'))
        x_mask = x_mask.unsqueeze(1).unsqueeze(2)
        ctx_freqs_ids = (torch.zeros_like(frame_start).unsqueeze(1) +
                         torch.arange(context.size(2), device=tokens.device).unsqueeze(0))
        ctx_freqs_cis = self.attn_freqs_cis[ctx_freqs_ids]
        ctx_mask = torch.zeros((context.size(0), context.size(2)), device=tokens.device)
        ctx_mask.masked_fill_(~sequence_mask(context_lengths), float('-inf'))
        ctx_mask = ctx_mask.unsqueeze(1).unsqueeze(2)

        output = self.ctx_block(
            expanded_phone, context, freqs_cis, ctx_freqs_cis, x_mask, ctx_mask)

        output = output * non_padding.unsqueeze(2)
        return output.transpose(1, 2)


class DurInputTransformerAdaptor(InputAdaptor):
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
        freqs_ids = phone_start.unsqueeze(1) + torch.arange(num_phones, device=tokens.device).unsqueeze(0)
        freqs_cis = self.attn_freqs_cis[freqs_ids]

        phone_mask = torch.zeros(*tokens.size(), device=tokens.device)
        phone_mask.masked_fill_(tokens == self.pad_token, float('-inf'))
        phone_mask = phone_mask.unsqueeze(1).unsqueeze(2)

        for layer in self.layers:
            h = layer(h, freqs_cis, mask=phone_mask)
        h = self.attn_norm(h)
        h = self.attn_output(h)
        non_padding = (tokens != self.pad_token).float()
        out = h * non_padding.unsqueeze(2)
        return out.transpose(1, 2)


class MMDiTInputAdaptor(InputAdaptor):
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
        return te, ce


class InputAdaptorProject(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.linear = nn.Linear(input_channels, output_channels)

    def forward(self, x, pad_val=0.):
        non_padding = (x.abs().sum(1, keepdim=True) > 0.).float()
        out = self.linear(x.transpose(1, 2)).transpose(1, 2)
        return out * non_padding + pad_val * (1. - non_padding)
