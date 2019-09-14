import torch
from torch import nn
from normal import NormalDistribution

class Encoder(nn.Module):
    def __init__(self, net, obs_dim, z_dim):
        super(Encoder, self).__init__()
        self.net = net
        self.img_dim = obs_dim
        self.z_dim = z_dim

    def forward(self, x):
        """
        :param x: observation
        :return: the parameters of distribution q(z|x)
        """
        return self.net(x).chunk(2, dim = 1) # first half is mean, second half is logvar

class Decoder(nn.Module):
    def __init__(self, net, z_dim, obs_dim):
        super(Decoder, self).__init__()
        self.net = net
        self.z_dim = z_dim
        self.obs_dim = obs_dim

    def forward(self, z):
        """
        :param z: sample from q(z|x)
        :return: reconstructed x
        """
        return self.net(z)

class Transition(nn.Module):
    def __init__(self, net, z_dim, u_dim, pertubations = False):
        super(Transition, self).__init__()
        self.net = net  # network to output the last layer before predicting A_t, B_t and o_t
        self.h_dim = self.net[-3].out_features
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.pertubations = pertubations

        if not pertubations:
            self.fc_A = nn.Sequential(
                nn.Linear(self.h_dim, self.z_dim * self.z_dim),
                nn.Sigmoid()
            )
        else:
            self.fc_A = nn.Sequential(
                nn.Linear(self.h_dim, self.z_dim * 2), # v_t and r_t
                nn.Sigmoid()
            )
        self.fc_B = nn.Linear(self.h_dim, self.z_dim * self.u_dim)
        self.fc_o = nn.Linear(self.h_dim, self.z_dim)

    def forward(self, z_bar_t, q_z_t, u_t): # Q_z_t is the distribution of z_bar_t
        """
        :param z_bar_t: the reference point
        :param Q_z_t: the distribution q(z|x)
        :param u_t: the action taken
        :return: the predicted q(z^_t+1 | z_t, z_bar_t, u_t)
        """
        h_t = self.net(z_bar_t)
        B_t = self.fc_B(h_t)
        o_t = self.fc_o(h_t)
        if not self.pertubations:
            A_t = self.fc_A(h_t)
            A_t = A_t.view(-1, self.z_dim, self.z_dim)
        else:
            v_t, r_t = self.fc_A(h_t).chunk(2, dim=1)
            v_t = torch.unsqueeze(v_t, dim=-1)
            r_t = torch.unsqueeze(r_t, dim=-2)
            A_t = torch.eye(self.z_dim).cuda() + torch.bmm(v_t, r_t)

        B_t = B_t.view(-1, self.z_dim, self.u_dim)

        mu_t = q_z_t.mean
        sigma_t = q_z_t.cov

        mean = A_t.bmm(mu_t.unsqueeze(-1)).squeeze(-1) + B_t.bmm(u_t.unsqueeze(-1)).squeeze(-1) + o_t
        cov = A_t.bmm(sigma_t.bmm(A_t.transpose(1, 2)))

        return NormalDistribution(mean, cov)

class PlanarEncoder(Encoder):
    def __init__(self, obs_dim = 1600, z_dim = 2):
        net = nn.Sequential(
            nn.Linear(obs_dim, 150),
            nn.BatchNorm1d(150),
            nn.ReLU(),

            nn.Linear(150, 150),
            nn.BatchNorm1d(150),
            nn.ReLU(),

            nn.Linear(150, 150),
            nn.BatchNorm1d(150),
            nn.ReLU(),

            nn.Linear(150, z_dim * 2)
        )
        super(PlanarEncoder, self).__init__(net, obs_dim, z_dim)

class PlanarDecoder(Decoder):
    def __init__(self, z_dim = 2, obs_dim = 1600):
        net = nn.Sequential(
            nn.Linear(z_dim, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),

            nn.Linear(200, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),

            nn.Linear(200, 1600),
            nn.Sigmoid()
        )
        super(PlanarDecoder, self).__init__(net, z_dim, obs_dim)

class PlanarTransition(Transition):
    def __init__(self, z_dim = 2, u_dim = 2, pertubations = False):
        net = nn.Sequential(
            nn.Linear(z_dim, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),

            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU()
        )
        super(PlanarTransition, self).__init__(net, z_dim, u_dim, pertubations)


CONFIG = {
    'planar': (PlanarEncoder, PlanarDecoder, PlanarTransition)
}

def load_config(name):
    return CONFIG['name']

__all__ = ['load_config']

# enc = PlanarEncoder()
# dec = PlanarDecoder()
# trans = PlanarTransition()
#
# x = torch.randn(size=(10, 1600))
# # print (x.size())
# mean, logvar = enc(x)
# # print (logvar.size())
# x_recon = dec(mean)
# # print (x_recon.size())
#
# q_z_t = NormalDistribution(mean, torch.diag_embed(logvar.exp()))
# u_t = torch.randn(size=(10, 2))
# z_t_1 = trans(mean, q_z_t, u_t)
# print (z_t_1.cov[0])

