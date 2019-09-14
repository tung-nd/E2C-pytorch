import numpy as np
import torch

# class NormalDistribution:
#     def __init__(self, mean, logvar, A_t = None):
#         """
#         :param mean: mean of the distribution q(z_t | x_t) or q(z^_t+1 | z_t, A_t)
#         :param logvar: log variance of q(z_t | x_t)
#         :param A_t:
#         """
#         self.mean = mean
#         self.logvar = logvar
#         self.var = self.logvar.exp()
#         self.A_t = A_t
#         self.cov = self.compute_cov()
#
#     def compute_cov(self):
#         """
#         :return: the covariance matrix of q(z_t | x_t) (if A_t is none) or of q(z^_{t+1} | z_t, A_t)
#         """
#         cov = torch.diag_embed(self.var)
#         if self.A_t is None:
#             return cov
#         else:

class NormalDistribution:
    def __init__(self, mean, cov):
        self.mean = mean
        if len(cov.size()) == 2:
            cov = torch.diag_embed(cov)
        self.cov = cov