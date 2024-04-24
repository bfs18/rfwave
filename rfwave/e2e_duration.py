import torch
import torch.nn as nn
import math


from rfwave.attention import (
    sequence_mask, score_mask, _get_start, precompute_freqs_cis, get_pos_embed_indices)
from rfwave.input import TransformerBlock, ModelArgs


class ExpScale(nn.Module):
    def __init__(self, token_dur_model):
        super().__init__()
        self.token_dur_model = token_dur_model
        dim = self.token_dur_model.dim
        self.output_proj = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, 1))

    def forward(self, x, num_tokens, ref_length):
        token_out = self.token_dur_model.forward(x, num_tokens, ref_length)
        mask = sequence_mask(num_tokens)
        token_out = token_out * mask.unsqueeze(-1).float()
        avg_out = token_out.sum(1) / num_tokens.unsqueeze(1)
        out = self.output_proj(avg_out)
        return out.squeeze(-1)


class PhnDur(nn.Module):
    def __init__(self, token_dur_model):
        super().__init__()
        self.token_dur_model = token_dur_model
        dim = self.token_dur_model.dim
        self.output_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, x, num_tokens, ref_length):
        out = self.token_dur_model.forward(x, num_tokens, ref_length)
        return self.output_proj(out).squeeze(-1)


class DurModel(nn.Module):
    def __init__(self, dim, num_layers, dropout=0.):
        super().__init__()
        params = ModelArgs(dim=dim, n_layers=num_layers, n_heads=8, dropout=dropout)
        self.dim = dim
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.pad_token = 0
        freqs_cis = precompute_freqs_cis(params.dim // params.n_heads, params.max_seq_len)
        self.register_buffer("attn_freqs_cis", freqs_cis, persistent=False)

        # init all weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_pos_embed(self, start, length, scale=1.):
        # attn_freqs_cis = self.attn_freqs_cis if self.training else self.attn_freqs_cis_eval
        attn_freqs_cis = self.attn_freqs_cis
        pos = get_pos_embed_indices(start, length, max_pos=attn_freqs_cis.size(0), scale=scale)
        return attn_freqs_cis[pos]

    def forward(self, x, num_tokens, ref_length):
        # pos_embed for token and ref
        token_mask = score_mask(num_tokens)
        ref_mask = score_mask(ref_length)
        zero_start = _get_start(x, None)
        token_freq_cis = self.get_pos_embed(zero_start, num_tokens.max())
        ref_freq_cis = self.get_pos_embed(zero_start, ref_length.max())
        ctx_mask = torch.cat([token_mask, ref_mask], dim=-1)
        ctx_freq_cis = torch.cat([token_freq_cis, ref_freq_cis], dim=1)

        h = x.transpose(-2, -1)
        for layer in self.layers:
            h = layer(h, ctx_freq_cis, mask=ctx_mask)
        token_out, _ = torch.split(
            h, [num_tokens.max(), ref_length.max()], dim=1)
        return token_out


class E2EDuration(nn.Module):
    def __init__(self, backbone, output_exp_scale, rectified_flow=False):
        super(E2EDuration, self).__init__()
        self.output_exp_scale = output_exp_scale
        if self.output_exp_scale:
            self.backbone = ExpScale(backbone)
        else:
            self.backbone = PhnDur(backbone)

    def forward(self, x, num_tokens, ref_length):
        return self.backbone(x, num_tokens, ref_length)
