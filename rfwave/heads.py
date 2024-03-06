import torch
import librosa
import numpy as np

from torch import nn
from torchaudio.functional.functional import _hz_to_mel, _mel_to_hz
from torch.nn import functional as F

from rfwave.spectral_ops import IMDCT, ISTFT, STFT
from rfwave.modules import symexp


class FourierHead(nn.Module):
    """Base class for inverse fourier modules."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class RawFFTHead(FourierHead):
    def __init__(self, n_fft: int, hop_length: int, padding: str = "center"):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.p = (n_fft - hop_length) // 2
        self.cp = n_fft // 2
        self.padding = padding
        assert padding == 'center'
        win = librosa.filters.get_window('hamm', 1024, fftbins=True)
        self.register_buffer('win', torch.from_numpy(win / np.sum(win)).to(torch.float32))

    def get_wave(self, S):
        S = S.to(torch.complex128)
        y = torch.fft.irfft(S.transpose(1, 2), n=self.n_fft)
        y = y[..., self.p: y.size(-1) - self.p] / self.win[self.p: y.size(-1) - self.p]
        y = y.reshape(S.size(0), -1)
        y = y[:, self.cp - self.p: y.size(-1) - self.cp + self.p]
        y = y.to(torch.float32)
        return y

    def get_spec(self, audio):
        audio = audio.to(torch.float64)
        yp = F.pad(audio, (self.cp, self.cp))
        yw = yp.unfold(1, self.n_fft, self.hop_length)
        yw = yw * self.win
        S = torch.fft.rfft(yw, n=self.n_fft)
        S = S.to(torch.complex64)
        return S.transpose(1, 2)


class RFSTFTHead(FourierHead):
    """
    ISTFT Head module for predicting STFT complex coefficients.

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                          the resolution of the input features.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, dim: int, n_fft: int, hop_length: int, win_length: int = None, padding: str = "same"):
        super().__init__()
        out_dim = n_fft + 2
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.padding = padding
        if win_length is None or win_length <= 0:
            win_length = n_fft
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=win_length, padding=padding)
        self.stft = STFT(n_fft=n_fft, hop_length=hop_length, win_length=win_length, padding=padding)

    def get_wave(self, S):
        audio = self.istft(S)
        return audio

    def get_spec(self, audio):
        S = self.stft(audio)
        return S


class ISTFTHead(FourierHead):
    """
    ISTFT Head module for predicting STFT complex coefficients.

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                          the resolution of the input features.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, dim: int, n_fft: int, hop_length: int, win_length: int = None, padding: str = "same"):
        super().__init__()
        out_dim = n_fft + 2
        self.n_fft = n_fft
        self.padding = padding
        self.out = torch.nn.Linear(dim, out_dim)
        if win_length is None or win_length <= 0:
            win_length = n_fft
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=win_length, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)  # safeguard to prevent excessively large magnitudes
        # wrapping happens here. These two lines produce real and imaginary value
        x = torch.cos(p)
        y = torch.sin(p)
        # recalculating phase here does not produce anything new
        # only costs time
        # phase = torch.atan2(y, x)
        # S = mag * torch.exp(phase * 1j)
        # better directly produce the complex value 
        S = mag * (x + 1j * y)
        audio = self.istft(S)
        return audio

    def get_feat(self, x):
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)  # safeguard to prevent excessively large magnitudes
        # wrapping happens here. These two lines produce real and imaginary value
        x = torch.cos(p)
        y = torch.sin(p)
        # recalculating phase here does not produce anything new
        # only costs time
        # phase = torch.atan2(y, x)
        # S = mag * torch.exp(phase * 1j)
        # better directly produce the complex value
        S = mag * (x + 1j * y)
        return S

    def get_wave(self, S):
        audio = self.istft(S)
        return audio
