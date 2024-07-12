import torch.nn as nn
import torch.nn.functional as F

from vector_quantize_pytorch import ResidualFSQ
from rfwave.models import ConvNeXtV2Block


class Quantizer(nn.Module):
    def __init__(self, feat_dim, reduce=2, dim=None, num_layers=4,
                 levels=(5, 5, 5, 5), num_quantizers=2):
        super().__init__()
        if dim is None:
            dim = feat_dim
        self.reduce = reduce
        layers = [nn.Conv1d(feat_dim, dim, kernel_size=reduce * 2, stride=reduce)]
        if num_layers >= 1:
            layers.extend([ConvNeXtV2Block(dim, dim * 4) for _ in range(num_layers)])
        self.enc = nn.Sequential(*layers)
        self.rvq = ResidualFSQ(dim=dim, levels=levels, num_quantizers=num_quantizers)
        self.out = nn.ConvTranspose1d(dim, feat_dim, kernel_size=reduce * 2, stride=reduce)

    def forward(self, mel):
        padding = 0
        if self.reduce > 1:
            padding = self.reduce - (mel.size(-1) % self.reduce)
            mel = F.pad(mel, (0, padding))
        mel = self.enc(mel)
        z, *_ = self.rvq(mel.transpose(1, 2))
        z = z.transpose(1, 2)
        z = self.out(z)
        if self.reduce > 1 and padding > 0:
            z = z[..., :-padding]
        return z
