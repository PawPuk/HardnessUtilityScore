"""Wrappers for datasets that allowing mapping between"""

import glob
import os
from PIL import Image
from typing import Tuple

import torch


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        data, label = self.dataset[idx]
        if self.transform:
            data = self.transform(data)
        return data, label, idx

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


class SyntheticImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.paths = sorted(glob.glob(os.path.join(root_dir, "*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        label = int(os.path.basename(self.paths[idx]).split('_')[1].split('.')[0])
        if self.transform:
            img = self.transform(img)
        return img, label


class AugmentedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, int]:
        # Get the original data and label from the subset
        data, label, _ = self.subset[idx]

        # Apply the transformations to the data
        if self.transform:
            data = self.transform(data)
        return data, label, idx

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


