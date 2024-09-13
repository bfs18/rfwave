import torch
import julius
import typing as tp

from rfwave.pqmf import PQMF


class SampleProcessor(torch.nn.Module):
    def project_sample(self, x: torch.Tensor):
        """Project the original sample to the 'space' where the diffusion will happen."""
        return x

    def return_sample(self, z: torch.Tensor):
        """Project back from diffusion space to the actual sample space."""
        return z


class MultiBandProcessor(SampleProcessor):
    """
    MultiBand sample processor. The input audio is splitted across
    frequency bands evenly distributed in mel-scale.

    Each band will be rescaled to match the power distribution
    of Gaussian noise in that band, using online metrics
    computed on the first few samples.

    Args:
        n_bands (int): Number of mel-bands to split the signal over.
        sample_rate (int): Sample rate of the audio.
        num_samples (int): Number of samples to use to fit the rescaling
            for each band. The processor won't be stable
            until it has seen that many samples.
        power_std (float or list/tensor): The rescaling factor computed to match the
            power of Gaussian noise in each band is taken to
            that power, i.e. `1.` means full correction of the energy
            in each band, and values less than `1` means only partial
            correction. Can be used to balance the relative importance
            of low vs. high freq in typical audio signals.
    """
    def __init__(self, n_bands: int = 8, sample_rate: float = 24_000,
                 num_samples: int = 10_000, power_std: tp.Union[float, tp.List[float], torch.Tensor] = 1.):
        super().__init__()
        self.n_bands = n_bands
        self.split_bands = julius.SplitBands(sample_rate, n_bands=n_bands)
        self.num_samples = num_samples
        self.power_std = power_std
        if isinstance(power_std, list):
            assert len(power_std) == n_bands
            power_std = torch.tensor(power_std)
        self.register_buffer('counts', torch.zeros(1))
        self.register_buffer('sum_x', torch.zeros(n_bands))
        self.register_buffer('sum_x2', torch.zeros(n_bands))
        self.register_buffer('sum_target_x2', torch.zeros(n_bands))
        self.counts: torch.Tensor
        self.sum_x: torch.Tensor
        self.sum_x2: torch.Tensor
        self.sum_target_x2: torch.Tensor

    @property
    def mean(self):
        mean = self.sum_x / self.counts
        return mean

    @property
    def std(self):
        std = (self.sum_x2 / self.counts - self.mean**2).clamp(min=0).sqrt()
        return std

    @property
    def target_std(self):
        target_std = self.sum_target_x2 / self.counts
        return target_std

    def project_sample(self, x: torch.Tensor):
        assert x.dim() == 3
        bands = self.split_bands(x)
        if self.counts.item() < self.num_samples:
            ref_bands = self.split_bands(torch.randn_like(x))
            self.counts += len(x)
            self.sum_x += bands.mean(dim=(2, 3)).sum(dim=1)
            self.sum_x2 += bands.pow(2).mean(dim=(2, 3)).sum(dim=1)
            self.sum_target_x2 += ref_bands.pow(2).mean(dim=(2, 3)).sum(dim=1)
        rescale = (self.target_std / self.std.clamp(min=1e-12)) ** self.power_std  # same output size
        bands = (bands - self.mean.view(-1, 1, 1, 1)) * rescale.view(-1, 1, 1, 1)
        return bands.sum(dim=0)

    def return_sample(self, x: torch.Tensor):
        assert x.dim() == 3
        bands = self.split_bands(x)
        rescale = (self.std / self.target_std) ** self.power_std
        bands = bands * rescale.view(-1, 1, 1, 1) + self.mean.view(-1, 1, 1, 1)
        return bands.sum(dim=0)


class PQMFProcessor(SampleProcessor):
    def __init__(self, subbands=4, taps=62, cutoff_ratio=0.142, beta=9.0):
        super().__init__()
        self.pqmf = PQMF(subbands, taps, cutoff_ratio, beta)
        self.register_buffer('mean_ema', torch.zeros([subbands]))
        self.register_buffer('var_ema', torch.ones([subbands]))
        self.update = True

    def project_sample(self, x: torch.Tensor):
        audio_subbands = self.pqmf.analysis(x)
        if self.training and self.update:
            audio_subbands_mean = [torch.mean(x.float()) for x in torch.unbind(audio_subbands, dim=1)]
            audio_subbands_var = [torch.var(x.float()) for x in torch.unbind(audio_subbands, dim=1)]
            self.mean_ema.lerp_(torch.stack(audio_subbands_mean).detach(), 0.01)
            self.var_ema.lerp_(torch.stack(audio_subbands_var).detach(), 0.01)
        audio_subbands = (audio_subbands - self.mean_ema.unsqueeze(-1)) / torch.sqrt(self.var_ema.unsqueeze(-1) + 1e-6)
        audio = self.pqmf.synthesis(audio_subbands)
        return audio

    def return_sample(self, x: torch.Tensor):
        x_subbands = self.pqmf.analysis(x)
        x_subbands = x_subbands * torch.sqrt(self.var_ema.unsqueeze(-1) + 1e-6) + self.mean_ema.unsqueeze(-1)
        x = self.pqmf.synthesis(x_subbands)
        return x


class STFTProcessor(SampleProcessor):
    def __init__(self, n_fft):
        super().__init__()
        self.register_buffer('mean_ema', torch.zeros([n_fft]))
        self.register_buffer('var_ema', torch.ones([n_fft]))
        self.update = True

    def project_sample(self, x: torch.Tensor):
        if self.training and self.update:
            mean = torch.mean(x.float(), dim=(0, 2))
            var = torch.var(x.float(), dim=(0, 2))
            self.mean_ema.lerp_(mean.detach(), 0.01)
            self.var_ema.lerp_(var.detach(), 0.01)
        return (x - self.mean_ema[None, :, None]) / torch.sqrt(self.var_ema[None, :, None] + 1e-6)

    def return_sample(self, x: torch.Tensor):
        x = x * torch.sqrt(self.var_ema[None, :, None] + 1e-6) + self.mean_ema[None, :, None]
        return x
