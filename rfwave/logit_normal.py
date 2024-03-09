import torch
from torch.distributions import Normal


class LogitNormal(torch.nn.Module):
    def __init__(self, mu=0., sigma=1.):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.normal = Normal(loc=mu, scale=sigma)

    def sample(self, sample_shape=torch.Size()):
        x = self.normal.sample(sample_shape=sample_shape)
        return torch.sigmoid(x)

    def inv_cdf(self, value):
        y = self.mu + self.sigma * self.normal.icdf(value)
        return torch.sigmoid(y)

