from torch import nn
from dataclasses import dataclass
from typing import Optional
from rfwave.models import ConvNeXtV2Block
from rfwave.attention import (
    Attention, CrossAttention, CrossAttentionWithPrior, FeedForward,
    MLP, ConvFeedForward, RMSNorm, apply_rotary_emb, get_pos_embed_indices,
    score_mask, precompute_freqs_cis, score_mask_from_bool_mask, modulate, _get_start)
from rfwave.dataset import get_num_tokens

import torch
import math
import numpy as np


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
    hidden_dim: Optional[int] = None
    multiple_of: int = 256  # MLP hidden layer size will be multiple of
    norm_eps: float = 1e-6
    max_seq_len: int = 8192
    dropout: float = 0.0
    qk_norm: bool = True


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(dim=args.dim, num_heads=args.n_heads, qkv_bias=False, qk_norm=args.qk_norm,
                                   attn_drop=args.dropout, proj_drop=args.dropout, norm_layer=RMSNorm)
        # self.feed_forward = MLP(dim=args.dim, hidden_dim=args.hidden_dim, drop=args.dropout,
        #                         act_layer=lambda: nn.GELU(approximate="tanh"))
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=args.hidden_dim, drop=args.dropout, multiple_of=args.multiple_of)
        # self.feed_forward = ConvFeedForward(
        #     dim=args.dim, hidden_dim=args.hidden_dim, drop=args.dropout, multiple_of=args.multiple_of)
        # self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.attention_norm = nn.LayerNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.LayerNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cis, mask):
        h = x + self.attention.forward(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class CharInputAdaptor(InputAdaptor):
    def __init__(self, embedding_dim, vocab_size, n_attn_layers=4, n_conv_layers=4,
                 dropout=0., dilation=1):
        super().__init__()
        params = ModelArgs(dim=embedding_dim, n_layers=n_attn_layers, n_heads=8, dropout=dropout)
        self.dim = embedding_dim

        self.tok_embeddings = nn.Embedding(vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        # self.attn_norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.attn_norm = nn.LayerNorm(params.dim, eps=params.norm_eps)
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

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight') or pn.endswith('fc2.weight') or pn.endswith('w3.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_pos_embed(self, start, length, scale=1., eval_theta_rescale=False):
        if eval_theta_rescale:
            attn_freqs_cis = self.attn_freqs_cis if self.training else self.attn_freqs_cis_eval
        else:
            attn_freqs_cis = self.attn_freqs_cis
        pos = get_pos_embed_indices(start, length, max_pos=attn_freqs_cis.size(0), scale=scale)
        return attn_freqs_cis[pos]

    def forward_phone(self, tokens: torch.Tensor, phone_start: torch.Tensor):
        _bsz, num_phones = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        freqs_cis = self.get_pos_embed(phone_start, num_phones)
        # phone_mask = score_mask_from_bool_mask(tokens == self.pad_token)
        phone_mask = score_mask(get_num_tokens(tokens, self.pad_token))

        for layer in self.layers:
            h = layer(h, freqs_cis, mask=phone_mask)
        h = self.attn_norm(h)
        h = self.attn_output(h)
        return h

    def expand(self, encoded_phone, lengths):
        l = torch.max(lengths.sum(1))
        out = torch.zeros([encoded_phone.size(0), l, encoded_phone.size(2)], device=encoded_phone.device)
        for i, (phn, l) in enumerate(zip(encoded_phone, lengths)):
            out[i, :l.sum()] = phn.repeat_interleave(l, dim=0)
        return out

    def forward(self, tokens: torch.Tensor, token_frames: torch.Tensor,
                phone_start: torch.Tensor, frame_start: torch.Tensor, *args):
        encoded_phone = self.forward_phone(tokens, phone_start)
        expanded_phone = self.expand(encoded_phone, token_frames)
        num_frames = torch.sum(token_frames, dim=1).long()
        freqs_cis = self.get_pos_embed(frame_start, num_frames.max())
        rpt = expanded_phone.size(-1) // freqs_cis.size(-1)
        freqs_cis = freqs_cis.repeat_interleave(rpt, dim=-1)
        expanded_phone = apply_rotary_emb(expanded_phone, freqs_cis)
        output = self.convnext(expanded_phone.transpose(1, 2))
        output = self.output(output.transpose(1, 2))
        return output.transpose(1, 2)


class CrossAttTransformerBlock(nn.Module):
    def __init__(self, layer_id, args: ModelArgs, modulate=False):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(
            dim=args.dim, num_heads=args.n_heads, qkv_bias=False, qk_norm=args.qk_norm,
            attn_drop=args.dropout, proj_drop=args.dropout, norm_layer=RMSNorm)
        self.cross_attention = CrossAttention(
            dim=args.dim, num_heads=args.n_heads, qkv_bias=False, qk_norm=args.qk_norm,
            attn_drop=args.dropout, proj_drop=args.dropout, norm_layer=RMSNorm)
        # self.feed_forward = MLP(dim=args.dim, hidden_dim=args.hidden_dim, drop=args.dropout,
        #                         act_layer=lambda: nn.GELU(approximate="tanh"))
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=args.hidden_dim, drop=args.dropout, multiple_of=args.multiple_of)
        # self.feed_forward = ConvFeedForward(
        #     dim=args.dim, hidden_dim=args.hidden_dim, drop=args.dropout, multiple_of=args.multiple_of)
        self.layer_id = layer_id
        if not modulate:
            self.adaLN_modulation = None
            # self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
            # self.cross_attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
            # self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
            self.attention_norm = nn.LayerNorm(args.dim, eps=args.norm_eps)
            self.cross_attention_norm = nn.LayerNorm(args.dim, eps=args.norm_eps)
            self.ffn_norm = nn.LayerNorm(args.dim, eps=args.norm_eps)
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(self.dim, 9 * self.dim, bias=True))
            self.attention_norm = nn.LayerNorm(args.dim, elementwise_affine=False, eps=args.norm_eps)
            self.cross_attention_norm = nn.LayerNorm(args.dim, elementwise_affine=False, eps=args.norm_eps)
            self.ffn_norm = nn.LayerNorm(args.dim, elementwise_affine=False, eps=args.norm_eps)

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
        args = ModelArgs(dim, n_heads=1, multiple_of=128)
        self.dim = dim
        self.ctx_proj = nn.Conv1d(ctx_dim, args.dim, 1) if ctx_dim != args.dim else nn.Identity()
        self.align_attn = CrossAttentionWithPrior(
            dim=args.dim, num_heads=args.n_heads, qkv_bias=False, qk_norm=args.qk_norm,
            attn_drop=args.dropout, proj_drop=args.dropout, norm_layer=RMSNorm)
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
        self.apply(_basic_init)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, context, x_freqs_cis, c_freqs_cis, c_mask, mod_c, attn_prior):
        context = self.ctx_proj(context).transpose(1, 2)
        rpt = self.dim // x_freqs_cis.size(2)
        x_freqs_cis = x_freqs_cis.repeat_interleave(rpt, dim=-1)
        c_freqs_cis = c_freqs_cis.repeat_interleave(rpt, dim=-1)
        shift_crs, scale_crs, gate_crs = self.adaLN_modulation(mod_c).chunk(3, dim=1)
        h, score = self.align_attn(
            modulate(self.cross_attention_norm(x), shift_crs, scale_crs),
            context, x_freqs_cis, c_freqs_cis, c_mask, attn_prior=attn_prior)
        h = x + (gate_crs.unsqueeze(1) * h)
        return h, score


class ContextBlock(nn.Module):
    def __init__(self, args: ModelArgs, ctx_dim, n_attn_layers=4, modulate=False):
        super().__init__()
        self.num_layers = n_attn_layers
        self.ctx_proj = nn.Conv1d(ctx_dim, args.dim, 1) if ctx_dim != args.dim else nn.Identity()
        self.blocks = nn.ModuleList([CrossAttTransformerBlock(i, args, modulate=modulate)
                                     for i in range(n_attn_layers)])
        if not modulate:
            # self.attn_norm = RMSNorm(args.dim, eps=args.norm_eps)
            self.attn_norm = nn.LayerNorm(args.dim, eps=args.norm_eps)
            self.adaLN_modulation = None
        else:
            self.attn_norm = nn.LayerNorm(args.dim, elementwise_affine=False, eps=args.norm_eps)
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(args.dim, 2 * args.dim, bias=True))
        self.attn_output = nn.Linear(args.dim, args.dim, bias=False)
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
            if pn.endswith('proj.weight') or pn.endswith('fc2.weight') or pn.endswith('w3.weight'):  # attention output weights
                torch.nn.init.trunc_normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.num_layers))

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
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

        for layer in self.blocks:
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
        params = ModelArgs(dim=embedding_dim, n_layers=n_attn_layers, n_heads=8, dropout=dropout)
        self.dim = embedding_dim
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        # self.attn_norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.attn_norm = nn.LayerNorm(params.dim, eps=params.norm_eps)
        self.attn_output = nn.Linear(params.dim, params.dim, bias=False)
        self.ctx_proj = nn.Sequential(
            nn.Conv1d(ctx_dim, params.dim, 1),
            *[ConvNeXtV2Block(params.dim, params.dim*3) for _ in range(n_conv_layers)])
        self.ctx_attn = ContextBlock(
            params, ctx_dim=params.dim, n_attn_layers=n_ctx_layers)
        self.pad_token = 0

        # some useful precompute for the RoPE relative positional embeddings
        freqs_cis = precompute_freqs_cis(params.dim // params.n_heads, params.max_seq_len)
        self.register_buffer("attn_freqs_cis", freqs_cis, persistent=False)
        freqs_cis = precompute_freqs_cis(params.dim // params.n_heads, params.max_seq_len, theta_rescale_factor=8.)
        self.register_buffer("attn_freqs_cis_eval", freqs_cis, persistent=False)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        self.layers.apply(_basic_init)
        for pn, p in self.layers.named_parameters():
            if pn.endswith('proj.weight') or pn.endswith('fc2.weight') or pn.endswith('w3.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers))
        self.ctx_proj.apply(_basic_init)
        nn.init.trunc_normal_(self.tok_embeddings.weight, std=0.02)
        nn.init.trunc_normal_(self.attn_output.weight, std=0.02)

    def get_pos_embed(self, start, length, scale=1., eval_theta_rescale=False):
        # phone theta_rescale and no expand theta_rescale performs better at evaluation
        if eval_theta_rescale:
            attn_freqs_cis = self.attn_freqs_cis if self.training else self.attn_freqs_cis_eval
        else:
            attn_freqs_cis = self.attn_freqs_cis
        pos = get_pos_embed_indices(start, length, max_pos=attn_freqs_cis.size(0), scale=scale)
        return attn_freqs_cis[pos]

    def forward_phone(self, tokens: torch.Tensor, phone_start: torch.Tensor):
        _bsz, num_phones = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        freqs_cis = self.get_pos_embed(phone_start, num_phones)
        # phone_mask = score_mask_from_bool_mask(tokens == self.pad_token)
        phone_mask = score_mask(get_num_tokens(tokens, self.pad_token))

        for layer in self.layers:
            h = layer(h, freqs_cis, mask=phone_mask)

        h = self.attn_norm(h)
        h = self.attn_output(h)
        return h

    def expand(self, encoded_phone, lengths):
        l = torch.max(lengths.sum(1))
        out = torch.zeros([encoded_phone.size(0), l, encoded_phone.size(2)], device=encoded_phone.device)
        for i, (phn, l) in enumerate(zip(encoded_phone, lengths)):
            out[i, :l.sum()] = phn.repeat_interleave(l, dim=0)
        return out

    def forward(self, tokens: torch.Tensor, token_frames: torch.Tensor,
                phone_start: torch.Tensor, frame_start: torch.Tensor,
                context: torch.Tensor, context_lengths: torch.Tensor, *args, **kwargs):
        # context: [b, c, t]
        encoded_phone = self.forward_phone(tokens, phone_start)
        expanded_phone = self.expand(encoded_phone, token_frames)
        non_padding = (expanded_phone.abs().sum(2) > 0.).type_as(expanded_phone)
        num_frames = torch.sum(token_frames, dim=1).long()

        freqs_cis = self.get_pos_embed(frame_start, num_frames.max())
        x_mask = score_mask_from_bool_mask(non_padding == 0)
        ctx_freqs_cis = self.get_pos_embed(torch.zeros_like(frame_start), context.size(2))
        ctx_mask = score_mask(context_lengths)

        context = self.ctx_proj(context)
        output = self.ctx_attn(
            expanded_phone, context, freqs_cis, ctx_freqs_cis, x_mask, ctx_mask)

        return output.transpose(1, 2)


class Ctx2CharInputAdaptor(InputAdaptor):
    def __init__(self, embedding_dim, vocab_size, ctx_dim,
                 n_attn_layers=4, n_conv_layers=4, n_ctx_layers=4,
                 dropout=0., drop_ctx=0.5):
        super().__init__()
        params = ModelArgs(dim=embedding_dim, n_layers=n_attn_layers, n_heads=8, dropout=dropout)
        self.dim = embedding_dim
        self.drop_ctx = drop_ctx
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        # self.attn_norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.attn_norm = nn.LayerNorm(params.dim, eps=params.norm_eps)
        self.attn_output = nn.Linear(params.dim, params.dim, bias=False)
        self.ctx_proj = nn.Sequential(
            nn.Conv1d(ctx_dim, params.dim, 1),
            *[ConvNeXtV2Block(params.dim, params.dim*3) for _ in range(n_conv_layers)])
        self.ctx_attn = ContextBlock(
            params, ctx_dim=params.dim*2, n_attn_layers=n_ctx_layers)
        self.pad_token = 0

        # some useful precompute for the RoPE relative positional embeddings
        freqs_cis = precompute_freqs_cis(params.dim // params.n_heads, params.max_seq_len)
        self.register_buffer("attn_freqs_cis", freqs_cis, persistent=False)
        freqs_cis = precompute_freqs_cis(params.dim // params.n_heads, params.max_seq_len, theta_rescale_factor=8.)
        self.register_buffer("attn_freqs_cis_eval", freqs_cis, persistent=False)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        self.layers.apply(_basic_init)
        for pn, p in self.layers.named_parameters():
            if pn.endswith('proj.weight') or pn.endswith('fc2.weight') or pn.endswith('w3.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers))
        self.ctx_proj.apply(_basic_init)
        nn.init.trunc_normal_(self.tok_embeddings.weight, std=0.02)
        nn.init.trunc_normal_(self.attn_output.weight, std=0.02)

    def get_pos_embed(self, start, length, scale=1., eval_theta_rescale=False):
        # phone theta_rescale and no expand theta_rescale performs better at evaluation
        if eval_theta_rescale:
            attn_freqs_cis = self.attn_freqs_cis if self.training else self.attn_freqs_cis_eval
        else:
            attn_freqs_cis = self.attn_freqs_cis
        pos = get_pos_embed_indices(start, length, max_pos=attn_freqs_cis.size(0), scale=scale)
        return attn_freqs_cis[pos]

    def forward_phone(self, tokens: torch.Tensor, phone_start: torch.Tensor):
        _bsz, num_phones = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        freqs_cis = self.get_pos_embed(phone_start, num_phones)
        # phone_mask = score_mask_from_bool_mask(tokens == self.pad_token)
        phone_mask = score_mask(get_num_tokens(tokens, self.pad_token))

        for layer in self.layers:
            h = layer(h, freqs_cis, mask=phone_mask)

        h = self.attn_norm(h)
        h = self.attn_output(h)
        return h

    def expand(self, encoded_phone, lengths):
        l = torch.max(lengths.sum(1))
        out = torch.zeros([encoded_phone.size(0), l, encoded_phone.size(2)], device=encoded_phone.device)
        for i, (phn, l) in enumerate(zip(encoded_phone, lengths)):
            out[i, :l.sum()] = phn.repeat_interleave(l, dim=0)
        return out

    def forward_ctx(self, context: torch.Tensor, ctx_tokens: torch.Tensor, ctx_token_frames: torch.Tensor):
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
        non_padding = (expanded_phone.abs().sum(2) > 0.).type_as(expanded_phone)
        num_frames = torch.sum(token_frames, dim=1).long()

        freqs_cis = self.get_pos_embed(frame_start, num_frames.max())
        x_mask = score_mask_from_bool_mask(non_padding == 0)
        ctx_freqs_cis = self.get_pos_embed(torch.zeros_like(frame_start), context.size(2))
        ctx_mask = score_mask(context_lengths)

        context = self.forward_ctx(context, ctx_tokens, ctx_token_frames)
        output = self.ctx_attn(
            expanded_phone, context, freqs_cis, ctx_freqs_cis, x_mask, ctx_mask)

        return output.transpose(1, 2)


class DurInputAdaptor(InputAdaptor):
    def __init__(self, embedding_dim, vocab_size, n_attn_layers=4, dropout=0.):
        super().__init__()
        params = ModelArgs(dim=embedding_dim, n_layers=n_attn_layers, n_heads=8, dropout=dropout)
        self.dim = embedding_dim

        self.tok_embeddings = nn.Embedding(vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        # self.attn_norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.attn_norm = nn.LayerNorm(params.dim, eps=params.norm_eps)
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
            if pn.endswith('proj.weight') or pn.endswith('fc2.weight') or pn.endswith('w3.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_pos_embed(self, start, length, scale=1.):
        attn_freqs_cis = self.attn_freqs_cis if self.training else self.attn_freqs_cis_eval
        pos = get_pos_embed_indices(start, length, max_pos=attn_freqs_cis.size(0), scale=scale)
        return attn_freqs_cis[pos]

    def forward(self, tokens: torch.Tensor):
        _bsz, num_phones = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        phone_start = torch.zeros([_bsz], dtype=torch.long, device=h.device)

        freqs_cis = self.get_pos_embed(phone_start, num_phones)
        # phone_mask = score_mask_from_bool_mask(tokens == self.pad_token)
        phone_mask = score_mask(get_num_tokens(tokens, self.pad_token))

        for layer in self.layers:
            h = layer(h, freqs_cis, mask=phone_mask)
        h = self.attn_norm(h)
        h = self.attn_output(h)
        return h.transpose(1, 2)


class E2ECtxCharInputAdaptor(InputAdaptor):
    def __init__(self, embedding_dim, vocab_size, ctx_dim, num_layers=4):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.ctx_proj = nn.Conv1d(ctx_dim, embedding_dim, kernel_size=1)
        self.tok_blocks = nn.Sequential(*[ConvNeXtV2Block(embedding_dim, embedding_dim*3) for _ in range(num_layers)])
        self.ctx_blocks = nn.Sequential(*[ConvNeXtV2Block(embedding_dim, embedding_dim*3) for _ in range(num_layers)])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

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

    def forward(self, x, padding_val=0.):
        non_padding = (x.abs().sum(1, keepdim=True) > 0.).type_as(x)
        out = self.linear(x.transpose(1, 2)).transpose(1, 2)
        return out * non_padding + padding_val * (1. - non_padding)


if __name__ == '__main__':
    from rfwave.dataset import DataConfig, TTSCtxDatasetSegment, tts_ctx_collate_segment
    from torch.utils.data import DataLoader
    from rfwave.feature_extractors import MelSpectrogramFeatures

    input_adaptor = CtxCharInputAdaptor(
        embedding_dim=512, vocab_size=78, ctx_dim=100)

    cfg = DataConfig(
        filelist_path="/Users/liupeng/wd_disk/dataset/LJSpeech-1.1/synta_filelist.valid",
        sampling_rate=22050,
        num_samples=65280,
        batch_size=8,
        num_workers=0,
        cache=True,
        task="tts",
        hop_length=256,
        padding="center",
        phoneset="/Users/liupeng/wd_disk/dataset/LJSpeech-1.1/synta_phoneset.th",
    )
    dataset = TTSCtxDatasetSegment(cfg, train=False)
    dataloader = DataLoader(dataset, collate_fn=tts_ctx_collate_segment, batch_size=cfg.batch_size)
    batch = next(iter(dataloader))
    phone_info = batch[1]
    mel_extractor = MelSpectrogramFeatures(22050, n_fft=1024, hop_length=256, n_mels=100)
    phone_info[4] = mel_extractor(phone_info[4])
    out = input_adaptor(*phone_info)
    print(out.shape)
