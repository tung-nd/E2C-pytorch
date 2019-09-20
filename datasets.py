import os
from os import path
from PIL import Image
import numpy as np
import json
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle
import torch

torch.set_default_dtype(torch.float64)

class PlanarDataset(Dataset):
    width = 40
    height = 40
    action_dim = 2

    def __init__(self, dir):
        self.dir = dir
        with open(path.join(dir, 'data.json')) as f:
            self._data = json.load(f)
        self._process()

    def __len__(self):
        return len(self._data['samples'])

    def __getitem__(self, index):
        return self._processed[index]

    @staticmethod
    def _process_image(img):
        return ToTensor()((img.convert('L').
                           resize((PlanarDataset.width,
                                   PlanarDataset.height))))

    def _process(self):
        preprocessed_file = os.path.join(self.dir, 'processed.pkl')
        if not os.path.exists(preprocessed_file):
            processed = []
            for sample in tqdm(self._data['samples'], desc='processing data'):
                before = Image.open(os.path.join(self.dir, sample['before']))
                after = Image.open(os.path.join(self.dir, sample['after']))

                processed.append((self._process_image(before),
                                  np.array(sample['control']),
                                  self._process_image(after)))

            with open(preprocessed_file, 'wb') as f:
                pickle.dump(processed, f)
            self._processed = processed
        else:
            with open(preprocessed_file, 'rb') as f:
                self._processed = pickle.load(f)

class GymPendulumDatasetV2(Dataset):
    width = 48 * 2
    height = 48
    action_dim = 1

    def __init__(self, dir):
        self.dir = dir
        with open(path.join(dir, 'data.json')) as f:
            self._data = json.load(f)
        self._process()

    def __len__(self):
        return len(self._data['samples'])

    def __getitem__(self, index):
        return self._processed[index]

    @staticmethod
    def _process_image(img):
        return ToTensor()((img.convert('L').
                           resize((GymPendulumDatasetV2.width,
                                   GymPendulumDatasetV2.height))))

    def _process(self):
        preprocessed_file = os.path.join(self.dir, 'processed.pkl')
        if not os.path.exists(preprocessed_file):
            processed = []
            for sample in tqdm(self._data['samples'], desc='processing data'):
                before = Image.open(os.path.join(self.dir, sample['before']))
                after = Image.open(os.path.join(self.dir, sample['after']))

                processed.append((self._process_image(before),
                                  np.array(sample['control']),
                                  self._process_image(after)))

            with open(preprocessed_file, 'wb') as f:
                pickle.dump(processed, f)
            self._processed = processed
        else:
            with open(preprocessed_file, 'rb') as f:
                self._processed = pickle.load(f)