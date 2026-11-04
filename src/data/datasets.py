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
        self._images = [Image.open(p).convert('RGB') for p in self.paths]
        self._labels = [int(os.path.basename(p).split('_')[1].split('.')[0]) for p in self.paths]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        img = self._images[idx]
        if self.transform:
            img = self.transform(img)
        return img, self._labels[idx]


class TinyImageNetValDataset(torch.utils.data.Dataset):
    """Custom Dataset for Tiny ImageNet validation set, which uses a flat folder + text file."""

    def __init__(self, root_dir: str, transform=None):
        self.transform = transform
        self.image_dir = os.path.join(root_dir, 'val', 'images')
        self.annotations_file = os.path.join(root_dir, 'val', 'val_annotations.txt')

        # Convert the WordNet IDs (e.g., 'n01443537') to integers 0-199 to match those in training set.
        train_dir = os.path.join(root_dir, 'train')
        self.classes = sorted(os.listdir(train_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.data = []
        with open(self.annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                img_name = parts[0]
                class_name = parts[1]
                self.data.append((img_name, self.class_to_idx[class_name]))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        img_name, label = self.data[idx]
        img_path = os.path.join(self.image_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label
