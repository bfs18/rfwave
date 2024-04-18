import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from matplotlib import pylab as plt
from numba import jit

from rfwave.models import ConvNeXtV2Block
from rfwave.dataset import get_exp_length


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


class StandaloneAlignment(torch.nn.Module):
    def __init__(self, n_mel_channels, n_text_channels,
                 n_channels, num_layers=3, temperature=1.0):
        super(StandaloneAlignment, self).__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=3)
        self.log_softmax = torch.nn.LogSoftmax(dim=3)

        self.key_proj = nn.Sequential(
            nn.Conv1d(n_text_channels, n_channels, 1),
            *[ConvNeXtV2Block(n_channels, n_channels * 3) for _ in range(num_layers)])

        self.query_proj = nn.Sequential(
            nn.Conv1d(n_mel_channels, n_channels, 1),
            *[ConvNeXtV2Block(n_channels, n_channels * 3) for _ in range(num_layers)])

    def forward(self, queries, keys, mask=None, attn_prior=None):
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
        temp = 0.0005
        keys_enc = self.key_proj(keys)  # B x n_attn_dims x T2
        # Beware can only do this since query_dim = attn_dim = n_mel_channels
        queries_enc = self.query_proj(queries)

        # Gaussian Isotopic Attention
        # B x n_attn_dims x T1 x T2
        attn = (queries_enc[:, :, :, None] - keys_enc[:, :, None])**2

        # compute log-likelihood from gaussian
        eps = 1e-8
        attn = -temp * attn.sum(1, keepdim=True)
        if attn_prior is not None:
            attn = self.log_softmax(attn) + torch.log(attn_prior[:, None] + eps)
        if mask is not None:
            attn = attn + mask

        attn_logprob = attn.clone()
        attn = self.softmax(attn)  # softmax along T2
        return attn, attn_logprob


def standalone_compute_alignment_loss(attn, num_tokens, token_exp_scale, blank_logprob=-1):
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
    attn = F.pad(attn, (1, 0, 0, 0, 0, 0), value=blank_logprob)  # prob for blank.
    # attn = attn / attn.sum(dim=-1, keepdim=True)
    # log_prob = torch.log(attn.clamp_min_(1e-5))
    log_prob = F.log_softmax(attn, dim=-1)
    loss = F.ctc_loss(log_prob.transpose(1, 0), targets=target, zero_infinity=True,
                      input_lengths=length, target_lengths=num_tokens, blank=0)
    return loss


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


def beta_binomial_prior_distribution(phoneme_count, mel_count,
                                     scaling_factor=0.05):
    P = phoneme_count
    M = mel_count
    x = np.arange(0, P)
    mel_text_probs = []
    for i in range(1, M+1):
        a, b = scaling_factor*i, scaling_factor*(M+1-i)
        rv = betabinom(P - 1, a, b)
        mel_i_prob = rv.pmf(x)
        mel_text_probs.append(mel_i_prob)
    return torch.tensor(np.array(mel_text_probs))


if __name__ == '__main__':
    attn_ = np.load(sys.argv[1])
    attn = attn_.squeeze()
    save_plot('orig.png', attn)
    binarized = mas_width1(attn)
    save_plot('binarized.png', binarized)

