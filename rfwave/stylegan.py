
import torch
import numpy as np
from torch import nn


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
                 in_features,  # Number of input features.
                 out_features,  # Number of output features.
                 bias=True,  # Apply additive bias before the activation function?
                 lr_multiplier=1.,  # Learning rate multiplier.
                 bias_init=0,  # Initial value for the additive bias.
                 ):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = torch.mm(x, w.t())
        return x


class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim, num_ws, num_layers=3,
                 lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
                 w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track.
                 ):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.w_avg_beta = w_avg_beta
        map_fn = []
        for i in range(num_layers):
            map_fn.extend([nn.Linear(z_dim, z_dim), nn.SiLU()])
        map_fn.append(FullyConnectedLayer(z_dim, w_dim, lr_multiplier=lr_multiplier))
        self.map_fn = nn.Sequential(*map_fn)

        if w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, truncation_psi=1, truncation_cutoff=None):
        skip_w_avg_update = not self.training
        x = normalize_2nd_moment(z)
        x = self.map_fn(x)
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            self.w_avg.copy_(x.detach().float().mean(dim=0).lerp(self.w_avg.float(), self.w_avg_beta))

        if self.num_ws is not None:
            x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        if truncation_psi != 1:
            assert self.w_avg_beta is not None
            if self.num_ws is None or truncation_cutoff is None:
                x = self.w_avg.lerp(x, truncation_psi)
            else:
                x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x


class StyleGANNoise(nn.Module):
    def __init__(self, dim, use_noise=True):
        super().__init__()
        self.use_noise = use_noise
        if use_noise:
            self.dim = dim
            self.map_fn = MappingNetwork(dim, dim, num_ws=1)
            self.register_buffer('noise_const', torch.randn([8192]), persistent=False)  # max 40 s audio.
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))

    def forward(self, x, noise_mode='random'):
        assert noise_mode in ['random', 'const', 'none']

        if not self.use_noise:
            return x

        z = torch.randn((x.shape[0], self.dim), device=x.device)
        z = self.map_fn(z)[:, 0]
        noise = 0.
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, x.shape[2]], device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const[..., :x.size(2)] * self.noise_strength
        return x * z.unsqueeze(2) + noise
