import torch
import numpy as np
import torch.nn.functional as F


def unwrap(p, discont=None, dim=-1, *, period=2*np.pi):
    nd = p.ndim
    dd = torch.diff(p, dim=dim)
    if discont is None:
        discont = period/2
    slice1 = [slice(None, None)]*nd     # full slices
    slice1[dim] = slice(1, None)
    slice1 = tuple(slice1)
    interval_high = period / 2
    boundary_ambiguous = True
    interval_low = -interval_high
    ddmod = torch.remainder(dd - interval_low, period) + interval_low
    if boundary_ambiguous:
        ddmod[(ddmod == interval_low) & (dd > 0)] = interval_high
    ph_correct = ddmod - dd
    ph_correct[abs(dd) < discont] = 0.
    up = p.clone()
    up[slice1] = p[slice1] + ph_correct.cumsum(dim)
    return up


def compute_instantaneous_frequency(complex_spec):
    phi = torch.angle(complex_spec)
    delta_phi = torch.remainder(torch.diff(phi, dim=2), 2 * np.pi)
    uw_delta_phi = unwrap(delta_phi, dim=2)
    psi = torch.remainder(
        (uw_delta_phi[..., :-1] + uw_delta_phi[..., 1:]) / 2., 2 * np.pi)

    return psi


def compute_phase_loss(pred_psi, targ_psi, w=None):
    if w is None:
        t_pad_targ_psi = F.pad(targ_psi, (1, 1, 0, 0))
        f_pad_targ_psi = F.pad(targ_psi, (0, 0, 1, 1))
        s_tm1 = torch.remainder(t_pad_targ_psi[..., :-2] - targ_psi, 2 * np.pi).abs()
        s_tp1 = torch.remainder(t_pad_targ_psi[..., 2:] - targ_psi, 2 * np.pi).abs()
        s_fm1 = torch.remainder(f_pad_targ_psi[:, :-2] - targ_psi, 2 * np.pi).abs()
        s_fp1 = torch.remainder(f_pad_targ_psi[:, 2:] - targ_psi, 2 * np.pi).abs()
        s = s_tm1 + s_tp1 + s_fm1 + s_fp1
        w = (1 + s).mean() / (1 + s)
    loss = ((1 - torch.cos(pred_psi - targ_psi)) * w).mean()
    return loss


def compute_phase_error(pred_wave, targ_wave, stft_func):
    pred_spec = stft_func(pred_wave)
    targ_spec = stft_func(targ_wave)
    pred_if = compute_instantaneous_frequency(pred_spec)
    targ_if = compute_instantaneous_frequency(targ_spec)
    phase_loss = compute_phase_loss(pred_if, targ_if)
    return phase_loss
