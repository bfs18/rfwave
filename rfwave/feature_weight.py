import torch
import numpy as np


def get_idft_basis(n):
    time_ids = np.arange(0, n)
    freq_ids = np.arange(0, n)
    # DFT matrix
    ids = (freq_ids[:, None] * time_ids[None, :] / n)
    cmtx = np.exp(2j * np.pi * ids)
    return cmtx / n


def get_feature_weight(n, hs):
    m = get_idft_basis(n)
    m_real = m.real[:, :n // 2 + 1]
    m_real[:, 1:-1] *= 2
    m_imag = m.imag[:, :n // 2 + 1]
    m_imag[:, 1:-1] *= 2
    return torch.from_numpy(np.concatenate([m_real, -m_imag], axis=1).astype(np.float32))


def get_feature_weight2(n, hs):
    bl = np.ceil(n // 2 / hs)
    nf = 10
    fi = 3
    assert bl < fi < nf - bl
    window = torch.hann_window(n)  # win used in RFSTFTHead

    def _func(x):
        r, i = torch.chunk(x, 2, dim=1)
        c = r + 1j * i
        o = torch.istft(c, n, hop_length=hs, win_length=n, window=window, center=True)
        o = F.pad(o, [n//2, n//2])
        o = o.unfold(1, n, hs)
        o = o.transpose(1, 2)
        o = o[..., fi: fi + 1]
        return o.sum(dim=(0, 2))

    S = torch.rand([1, n+2, nf])
    S = S.requires_grad_(True)
    w = torch.autograd.functional.jacobian(_func, S)
    return w[:, 0, :, fi].detach()
