import torch
from torch import nn

from normal import *
from networks import *

torch.set_default_dtype(torch.float64)

class E2C(nn.Module):
    def __init__(self, obs_dim, z_dim, u_dim, env = 'planar'):
        super(E2C, self).__init__()
        enc, dec, trans = load_config(env)

        self.encoder = enc(obs_dim=obs_dim, z_dim=z_dim)
        self.decoder = dec(z_dim=z_dim, obs_dim=obs_dim)
        self.transition = trans(z_dim=z_dim, u_dim=u_dim)

    def encode(self, x):
        """
        :param x:
        :return: mean and log variance of q(z | x)
        """
        return self.encoder(x)

    def decode(self, z):
        """
        :param z:
        :return: bernoulli distribution p(x | z)
        """
        return self.decoder(z)

    def transition(self, z_bar, q_z, u):
        """
        :param z_bar:
        :param q_z:
        :param u:
        :return: samples z_hat_next and Q(z_hat_next)
        """
        return self.transition(z_bar, q_z, u)

    def reparam(self, mean, logvar):
        sigma = (logvar / 2).exp()
        epsilon = torch.randn_like(sigma)
        return mean + torch.mul(epsilon, sigma)

    def forward(self, x, u, x_next):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        q_z = NormalDistribution(mu, logvar)

        x_recon = self.decode(z)

        z_next, q_z_next_pred = self.transition(z, q_z, u)

        x_next_pred = self.decode(z_next)

        return x_recon, x_next_pred, q_z, q_z_next_pred

def compute_loss(x, x_next, q_z_next, x_recon, x_next_pred, q_z, q_z_next_pred):
    # lower-bound loss
    recon_term = -torch.mean(torch.sum(x * torch.log(1e-5 + x_recon) + (1 - x) * torch.log(1e-5 + 1 - x_recon), dim=1))
    pred_loss = -torch.mean(torch.sum(x_next * torch.log(1e-5 + x_next_pred) + (1 - x_next) * torch.log(1e-5 + 1 - x_next_pred), dim=1))

    kl_term = - 0.5 * torch.mean(torch.sum(1 + q_z.logvar - q_z.mean.pow(2) - q_z.logvar.exp(), dim = 1))

    lower_bound = recon_term + pred_loss + kl_term

    # consistency loss
    consis_term = NormalDistribution.KL_divergence(q_z_next_pred, q_z_next)
    return lower_bound + consis_term