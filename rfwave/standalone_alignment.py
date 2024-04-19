import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from matplotlib import pylab as plt
from numba import jit

from rfwave.models import ConvNeXtV2Block
from rfwave.dataset import get_exp_length
from rfwave.attention import (
    precompute_freqs_cis, get_pos_embed_indices, _get_start, _get_len, apply_rotary_emb)


def save_plot(fname, attn_map):
    plt.imshow(attn_map)
    plt.savefig(fname)


@jit(nopython=True)
def mas_width1(attn_map):
    """mas with hardcoded width=1"""
    # assumes mel x text
    opt = np.zeros_like(attn_map)
    attn_map = np.log(attn_map)
    attn_map[0, 1:] = -np.inf
    log_p = np.zeros_like(attn_map)
    log_p[0, :] = attn_map[0, :]
    prev_ind = np.zeros_like(attn_map, dtype=np.int64)
    for i in range(1, attn_map.shape[0]):
        for j in range(attn_map.shape[1]):  # for each text dim
            prev_log = log_p[i-1, j]
            prev_j = j

            if j-1 >= 0 and log_p[i-1, j-1] >= log_p[i-1, j]:
                prev_log = log_p[i-1, j-1]
                prev_j = j-1

            log_p[i, j] = attn_map[i, j] + prev_log
            prev_ind[i, j] = prev_j

    # now backtrack
    curr_text_idx = attn_map.shape[1]-1
    for i in range(attn_map.shape[0]-1, -1, -1):
        opt[i, curr_text_idx] = 1
        curr_text_idx = prev_ind[i, curr_text_idx]
    opt[0, curr_text_idx] = 1
    return opt


def compute_alignment_loss(attn, num_tokens, token_exp_scale, blank_prob=0.25):
    attn = attn.reshape(-1, *attn.shape[-2:])
    bsz = attn.shape[0]
    rpt = bsz // num_tokens.size(0)
    num_tokens = num_tokens.repeat_interleave(rpt, 0)
    token_exp_scale = token_exp_scale.repeat_interleave(rpt, 0)
    # length = torch.round(num_tokens * token_exp_scale).long()
    length = get_exp_length(num_tokens, token_exp_scale)
    target = torch.zeros([bsz, num_tokens.max()], device=attn.device, dtype=torch.long)
    for i, n_tok in enumerate(num_tokens):
        target[i, :n_tok] = torch.arange(1, n_tok + 1, device=attn.device)
    attn = F.pad(attn, (1, 0, 0, 0, 0, 0), value=blank_prob)  # prob for blank.
    attn = attn / attn.sum(dim=-1, keepdim=True)
    # float to compute loss.
    log_prob = torch.log(attn.clamp_min(1e-8)).float()
    loss = F.ctc_loss(log_prob.transpose(1, 0), targets=target, zero_infinity=True,
                      input_lengths=length, target_lengths=num_tokens, blank=0)
    return loss


def compute_attention_distill_loss(pred_attn, target_attn):
    # deal with band repeat.
    rpt = pred_attn.size(0) // target_attn.size(0)
    # float to compute loss.
    pred_attn, target_attn = pred_attn.float(), target_attn.float()
    target_attn = target_attn.repeat_interleave(rpt, 0)
    log_pred_attn = torch.log(pred_attn.clamp_min(1e-8))
    log_target_attn = torch.log(target_attn.clamp_min(1e-8))
    loss = F.kl_div(log_pred_attn, log_target_attn, reduction='none', log_target=True)  # , reduction='batchmean')
    loss = loss.sum(dim=-1).mean()
    return loss


class StandaloneAlignment(torch.nn.Module):
    def __init__(self, n_mel_channels, n_text_channels, n_channels,
                 num_layers=3, temperature=1.0, type='guassian', prior_strength=0.1):
        assert type in ['gaussian', 'dot_product']
        super(StandaloneAlignment, self).__init__()
        self.temperature = temperature
        self.scale = n_channels ** -0.5
        self.type = type
        self.prior_strength = prior_strength

        self.key_in = nn.Sequential(nn.Linear(n_text_channels, n_channels), nn.LayerNorm(n_channels))
        self.key_proj = nn.Sequential(*[ConvNeXtV2Block(n_channels, n_channels * 3) for _ in range(num_layers)])

        self.query_in = nn.Sequential(nn.Linear(n_mel_channels, n_channels), nn.LayerNorm(n_channels))
        self.query_proj = nn.Sequential(*[ConvNeXtV2Block(n_channels, n_channels * 3) for _ in range(num_layers)])

        self.register_buffer("freqs_cis", precompute_freqs_cis(n_channels, 8192), persistent=False)

    def get_pos_embed(self, start, length, scale=1.):
        # TODO: theta_rescale performs better at evaluation for dit vocoder.
        pos = get_pos_embed_indices(start, length, max_pos=self.freqs_cis.size(0), scale=scale)
        return self.freqs_cis[pos]

    def forward_guassian(self, queries, keys, token_exp_scale, mask=None, attn_prior=None):
        """Attention mechanism for radtts. Unlike in Flowtron, we have no
        restrictions such as causality etc, since we only need this during
        training.

        Args:
            queries (torch.tensor): B x C x T1 tensor (likely mel data)
            keys (torch.tensor): B x C2 x T2 tensor (text data)
            query_lens: lengths for sorting the queries in descending order
            mask (torch.tensor): uint8 binary mask for variable length entries
                                 (should be in the T2 domain)
        Output:
            attn (torch.tensor): B x 1 x T1 x T2 attention mask.
                                 Final dim T2 should sum to 1
        """
        start = _get_start(keys, None)
        keys_length = _get_len(keys, None)  # length is None
        queries_length = _get_len(queries, None)

        keys = self.key_in(keys.transpose(-2, -1))
        queries = self.query_in(queries.transpose(-2, -1))
        pos_a = np.sqrt(self.prior_strength)
        key_freq_cis = self.get_pos_embed(start, keys_length.max(), scale=token_exp_scale) * pos_a
        query_freq_cis = self.get_pos_embed(start, queries_length.max()) * pos_a
        # apply rotary emb before proj layer because it is designed for dot product attention,
        # while this is a Gaussian Isotopic Attention.
        # It is used as absolute positional embedding here.
        keys = apply_rotary_emb(keys, key_freq_cis)
        queries = apply_rotary_emb(queries, query_freq_cis)

        keys_enc = self.key_proj(keys.transpose(-2, -1))  # B x n_attn_dims x T2
        # Beware can only do this since query_dim = attn_dim = n_mel_channels
        queries_enc = self.query_proj(queries.transpose(-2, -1))

        # Gaussian Isotopic Attention
        # B x n_attn_dims x T1 x T2
        attn = (queries_enc[:, :, :, None] - keys_enc[:, :, None])**2

        # compute log-likelihood from gaussian
        attn = -attn.mean(1, keepdim=True)
        if mask is not None:
            attn = attn + mask
        attn = torch.softmax(attn.float() / self.temperature, dim=-1).type_as(attn)  # softmax along T2
        if attn_prior is not None:
            # attn = torch.exp(torch.log(attn.clamp_min(1e-8)) +
            #                  torch.log(attn_prior.unsqueeze(1).clamp_min(1e-8)))
            attn = attn + attn_prior * self.prior_strength
            attn = attn / attn.sum(dim=-1, keepdim=True)
        return attn

    def forward_dot_product(self, queries, keys, token_exp_scale, mask=None, attn_prior=None):
        # NOTE: attn_prior is not used. rotary positional embedding works as bias.
        keys = self.key_in(keys.transpose(-2, -1)).transpose(-2, -1)
        keys_enc = self.key_proj(keys)  # B x n_attn_dims x T2
        # Beware can only do this since query_dim = attn_dim = n_mel_channels
        queries = self.query_in(queries.transpose(-2, -1)).transpose(-2, -1)
        queries_enc = self.query_proj(queries)

        pos_a = np.sqrt(self.prior_strength)
        start = _get_start(keys_enc, None)
        keys_length = _get_len(keys_enc, None)  # length is None
        queries_length = _get_len(queries_enc, None)
        key_freq_cis = self.get_pos_embed(start, keys_length.max(), scale=token_exp_scale) * pos_a
        query_freq_cis = self.get_pos_embed(start, queries_length.max()) * pos_a
        keys_enc = apply_rotary_emb(keys_enc.transpose(-2, -1), key_freq_cis)
        queries_enc = apply_rotary_emb(queries_enc.transpose(-2, -1), query_freq_cis)

        queries_enc = queries_enc * self.scale
        attn = queries_enc @ keys_enc.transpose(-2, -1)
        attn = attn.unsqueeze(1)  # make attention shape consistent.
        if mask is not None:
            attn = attn + mask
        attn = torch.softmax(attn.float() / self.temperature, dim=-1).type_as(attn)
        if attn_prior is not None:
            # attn = torch.exp(torch.log(attn.clamp_min(1e-8)) +
            #                  torch.log(attn_prior.unsqueeze(1).clamp_min(1e-8)))
            attn = attn + attn_prior * self.prior_strength
            attn = attn / attn.sum(dim=-1, keepdim=True)
        return attn

    def forward(self, queries, keys, token_exp_scale, mask=None, attn_prior=None):
        if self.type == 'gaussian':
            return self.forward_guassian(queries, keys, token_exp_scale, mask, attn_prior)
        else:
            return self.forward_dot_product(queries, keys, token_exp_scale, mask, attn_prior)


def gaussian_prior(num_tokens, token_exp_scale, gamma=0.2, strength=1.):
    length = get_exp_length(num_tokens, token_exp_scale)
    ts = torch.arange(length.max(), device=num_tokens.device).unsqueeze(0) / length.unsqueeze(1)
    ns = torch.arange(num_tokens.max(), device=num_tokens.device).unsqueeze(0) / num_tokens.unsqueeze(1)
    prior = torch.exp(-(ts.unsqueeze(2) - ns.unsqueeze(1)) ** 2 / (2 * gamma ** 2))
    return prior * strength


class EmptyAlignmentBlock(torch.nn.Module):
    def __init__(self, dim, ctx_dim):
        super().__init__()
        self.dim = dim
        self.ctx_proj = nn.Conv1d(ctx_dim, dim, 1) if ctx_dim != dim else nn.Identity()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.dim, self.dim, bias=True))

    def forward(self, x, context, attn, mod_c):
        context = self.ctx_proj(context).transpose(1, 2)
        context_time_expanded = torch.bmm(attn.squeeze(1), context)
        gate = self.adaLN_modulation(mod_c)
        out = x + gate.unsqueeze(1) * context_time_expanded
        return out


if __name__ == '__main__':
    # attn_ = np.load(sys.argv[1])
    # attn = attn_.squeeze()
    # save_plot('orig.png', attn)
    # binarized = mas_width1(attn)
    # save_plot('binarized.png', binarized)
    num_tokens = torch.tensor([10, 17])
    token_exp_scale = torch.tensor([8.7, 7.3])
    prior = gaussian_prior(num_tokens, token_exp_scale, gamma=0.2)
    print(prior.shape)
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot
    pyplot.imshow(prior[0].numpy(), origin='lower', aspect='auto')
    pyplot.colorbar()
    pyplot.tight_layout()
    pyplot.show()
    pyplot.imshow(prior[1].numpy(), origin='lower', aspect='auto')
    pyplot.colorbar()
    pyplot.tight_layout()
    pyplot.show()
