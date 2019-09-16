import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader

from datasets import GymPendulumDatasetV2

np.random.seed(0)

dataset = GymPendulumDatasetV2('data/pendulum_markov')

# x, u, x_next = dataset[0]
# print (x.size())

loader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=16)