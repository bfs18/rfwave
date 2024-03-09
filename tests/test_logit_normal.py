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


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    N = 100000
    logit_normal1 = LogitNormal()
    samples1 = logit_normal1.sample(sample_shape=torch.Size([N]))
    logit_normal2 = LogitNormal(sigma=0.5)
    samples2 = logit_normal2.sample(sample_shape=torch.Size([N]))
    logit_normal3 = LogitNormal(mu=0.5)
    samples3 = logit_normal3.sample(sample_shape=torch.Size([N]))
    logit_normal4 = LogitNormal(mu=-0.5)
    samples4 = logit_normal4.sample(sample_shape=torch.Size([N]))
    plt.hist(samples1.numpy(), bins=100, density=True, alpha=0.3, label='mu=0, sigma=1')
    plt.hist(samples2.numpy(), bins=100, density=True, alpha=0.3, label='mu=0, sigma=0.5')
    plt.hist(samples3.numpy(), bins=100, density=True, alpha=0.3, label='mu=0.5, sigma=1')
    plt.hist(samples4.numpy(), bins=100, density=True, alpha=0.3, label='mu=-0.5, sigma=1')
    u = torch.linspace(0., 1., N)
    x = logit_normal1.inv_cdf(u)
    plt.hist(x.numpy(), bins=100, density=True, alpha=0.3, label='sample interval')
    plt.legend()
    plt.show()
    u = torch.linspace(0., 1., 11)
    x = logit_normal1.inv_cdf(u)
    print(x)
