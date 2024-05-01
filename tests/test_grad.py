import torch
import torchaudio
import numpy as np
import librosa
import torch.nn.functional as F

from rfwave.spectral_ops import STFT, ISTFT


stft = STFT(1024, 256, 1024, 'center')

y, sr = torchaudio.load('biaobeifemale4us_average_00001.wav')
S1 = librosa.stft(y.numpy(), n_fft=1024, hop_length=256, win_length=1024, center=True)
y1 = librosa.istft(S1, n_fft=1024, hop_length=256, win_length=1024, center=True)


def get_feature_weight(x):
    x = x.requires_grad_(True)
    r, i = torch.chunk(x, 2, dim=1)
    m = torch.sqrt(r ** 2 + i ** 2)
    angle = torch.angle(torch.complex(i, r))
    s = torch.sin(angle)
    c = torch.cos(angle)
    o = torch.cat([m, s, c], dim=1)
    # return o.sum((0, 2))
    return o


# x = torch.rand([3, 64 * 2, 72]).requires_grad_(True)
x = stft(y)
x = torch.cat([x.real, x.imag], dim=1)
# w = torch.autograd.functional.jacobian(get_feature_weight, x)
# print(w.shape, torch.norm(w))


def get_feature_weight2(x):
    x = x.requires_grad_(True)
    r, i = torch.chunk(x, 2, dim=1)
    m = torch.sqrt(r ** 2 + i ** 2)
    angle = torch.angle(torch.complex(i, r))
    s = torch.sin(angle)
    c = torch.cos(angle)
    o = torch.cat([m, s, c], dim=1)
    return o.sum((0, 2))


# w2 = torch.autograd.functional.jacobian(get_feature_weight2, x)
# print(w2.shape, torch.norm(w2))
#
# print(w.sum((0, 2)).shape)
# print(w.sum((3, 5)).permute(1, 0, 3, 2).shape)
#
# print(torch.allclose(w.sum((0, 2)), w2))
# print(torch.allclose(w.sum((3, 5)).permute(1, 0, 3, 2), w2))


def place_subband(sS, i, n_fft, num_bins, overlap, left_overlap, right_overlap):
    S = sS.new_zeros([sS.size(0), n_fft // 2 + overlap, sS.size(2), 2])
    rsS, isS = torch.chunk(sS, 2, dim=1)
    S[:, i * num_bins: (i + 1) * num_bins + overlap, :, 0] = rsS
    S[:, i * num_bins: (i + 1) * num_bins + overlap, :, 1] = isS
    S = S[:, left_overlap: S.size(1) - right_overlap + 1]
    return torch.cat([S[..., 0], S[..., 1]], dim=1)


n_fft = 1024
num_bins = 64
overlap = 32
left_overlap = 16
right_overlap = 16


window = torch.hann_window(n_fft)


def get_feature_loss2(x, bandwidth_i):
    def _func(x):
        # S = place_subband(x, bandwidth_i, n_fft, num_bins, overlap, left_overlap, right_overlap)
        r, i = torch.chunk(x, 2, dim=1)
        c = r + 1j * i
        o = torch.fft.irfft(c, n_fft, dim=1)
        print(o.shape)
        return o.sum(dim=(0, 2))
    x = x.requires_grad_(True)
    w = torch.autograd.functional.jacobian(_func, x)
    return w


def get_feature_loss3(x, fi):
    def _func(x):
        # S = place_subband(x, bandwidth_i, n_fft, num_bins, overlap, left_overlap, right_overlap)
        r, i = torch.chunk(x, 2, dim=1)
        c = r + 1j * i
        o = torch.istft(c, n_fft, hop_length=256, win_length=1024, window=window, center=True)
        o = F.pad(o, [512, 512])
        o = o.unfold(1, 1024, 256)
        o = o.transpose(1, 2)
        o = o[..., fi: fi + 1]
        print(o.shape)
        return o.sum(dim=(0, 2))
    x = x.requires_grad_(True)
    w = torch.autograd.functional.jacobian(_func, x)
    return w


torch.manual_seed(12345)
# S1 = torch.rand([3, (num_bins + overlap) * 2, 36])
S1 = torch.rand([3, 1026, 36])
w1 = get_feature_loss2(S1, 0)
# S2 = torch.rand([3, (num_bins + overlap) * 2, 36])
S2 = torch.rand([3, 1026, 36])
w2 = get_feature_loss2(S2, 0)
print(w1.shape, torch.allclose(S1, S2), torch.allclose(w1, w2))
print(torch.allclose(w1[:, 0, :, 0], w1[:, 2, :, 5]))
m1 = w1[:, 0, :, 0].detach().numpy()
m1_real, m1_imag = [x.numpy() for x in torch.chunk(w1[:, 0, :, 0], 2, dim=1)]
w0_ = get_feature_loss3(S1, 0)
w1_ = get_feature_loss3(S1, 1)
w2_ = get_feature_loss3(S2, 2)
w3_ = get_feature_loss3(S2, 3)
w4_ = get_feature_loss3(S2, 4)
w3__ = get_feature_loss3(S1, 3)
print('wave fold', w1.shape, torch.allclose(S1, S2), torch.allclose(w1_, w2_))

w0_np_ = w0_.numpy()[:, 0]
w1_np_ = w1_.numpy()[:, 0]
w2_np_ = w2_.numpy()[:, 0]
w3_np_ = w3_.numpy()[:, 0]
w4_np_ = w4_.numpy()[:, 0]
w3_np__ = w3__.numpy()[:, 0]
print('feature mtx close', np.allclose(w2_np_[..., 2], w3_np_[..., 3], atol=1e-5, rtol=1e-6))
print('feature mtx close', np.allclose(w3_np__[..., 3], w3_np_[..., 3], atol=1e-5, rtol=1e-6))
print('feature mtx close', np.allclose(w3_np_[..., 3][256:], w4_np_[..., 3][:-256], atol=1e-5, rtol=1e-6))


def get_idft_basis(n):
    time_ids = np.arange(0, n)
    freq_ids = np.arange(0, n)
    # DFT matrix
    ids = (freq_ids[:, None] * time_ids[None, :] / n)
    cmtx = np.exp(2j * np.pi * ids)
    return cmtx / n


m2 = get_idft_basis(1024)
m2_real = m2.real[:, :513]
m2_real[:, 1:-1] *= 2
m2_imag = m2.imag[:, :513]
m2_imag[:, 1:-1] *= 2

print(np.allclose(m1_real, m2_real))
print(np.allclose(m1_imag, -m2_imag))

feat_mtx1 = np.concatenate([m1_real, m1_imag], axis=1)
feat_mtx2 = w3_np_[..., 3]
r = feat_mtx2 / (feat_mtx1 + 1e-12)
print(torch.norm(torch.from_numpy(feat_mtx1)))
print(torch.norm(torch.from_numpy(feat_mtx2)))
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot
pyplot.imshow(feat_mtx1, aspect='auto', origin='lower', cmap='plasma')
pyplot.show()
pyplot.imshow(feat_mtx2, aspect='auto', origin='lower', cmap='plasma')
pyplot.show()
pyplot.imshow(r, aspect='auto', origin='lower', cmap='plasma')
pyplot.show()
