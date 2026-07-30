"""The data module: Provides two core Dataset subclasses, and the method for loading the data."""

from collections import Counter, defaultdict
import math
import os
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms

from src.config.config import get_config, ROOT
from src.data.datasets import AugmentedSubset, IndexedDataset, SyntheticImageDataset


def perform_data_augmentation(
        dataset: AugmentedSubset,
        dataset_name: str
) -> AugmentedSubset:
    """Applies data augmentation to the dataset. It firstly converts the images from Tensor to PIL to ensure the whole
    process is intact. This is useful in scenarios where we initially load the training dataset without applying data
    augmentation - load_dataset() with apply_augmentation=False. Specifically, in experiment2.py and experiment3.py"""
    mean = get_config(dataset_name)['mean']
    std = get_config(dataset_name)['std']

    data_augmentation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
    ])
    return AugmentedSubset(dataset, transform=data_augmentation)


def get_transform(
        apply_augmentation: bool,
        config: Dict[str, Union[int, float, List[int], List[float], List[str], Tuple[float, float, float]]]
) -> Tuple[transforms.Compose, transforms.Compose]:
    """For getting the transformation to the training and test sets."""
    if apply_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(config['mean'], config['std']),
        ])
    else:
        train_transform = transforms.ToTensor()

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config['mean'], config['std']),
    ])
    return train_transform, test_transform


def worker_init_fn(worker_id):
    """Set the seed for workers"""
    np.random.seed(42 + worker_id)
    random.seed(42 + worker_id)


def downsample_test_set_to_match_synthetic(
    test_set: IndexedDataset,
    synthetic_loader: DataLoader,
    random_seed: int = 42
) -> IndexedDataset:
    """
    Downsample the test set to match the class ratios of the synthetic set.
    Only under-sampling is performed; no oversampling.
    Classes absent in synthetic set are removed.

    Args:
        test_set: IndexedDataset (must yield (image, label, idx)).
        synthetic_loader: DataLoader yielding (images, labels, ...) from which
                          class frequencies are derived.
        random_seed: For reproducibility of sampling.

    Returns:
        A new IndexedDataset containing the downsampled test set.
    """
    np.random.seed(random_seed)

    # Collect synthetic labels
    synth_labels = []
    for _, labels, _ in synthetic_loader:
        synth_labels.extend(labels.cpu().numpy().tolist())
    synth_counts = Counter(synth_labels)
    total_synth = sum(synth_counts.values())

    # Compute proportions for classes present in synthetic_loader
    present_classes = [c for c in synth_counts if synth_counts[c] > 0]
    proportions = {c: synth_counts[c] / total_synth for c in present_classes}

    # Build mapping: class -> list of indices in test set
    test_indices_by_class = defaultdict(list)
    for _, label, i in test_set:
        test_indices_by_class[label].append(i)

    # Compute target counts per class
    target_counts = {}
    M = sum(len(test_indices_by_class[c]) for c in present_classes)
    max_desired_class_cardinality = 0
    for c in present_classes:
        desired_class_cardinality = int(round(proportions[c] * M))
        if desired_class_cardinality > max_desired_class_cardinality:
            max_desired_class_cardinality = desired_class_cardinality

    for c in present_classes:
        desired_class_cardinality = int(round(proportions[c] * M))
        current_class_cardinality = len(test_indices_by_class[c])
        # !!!!!!! IMPORTANT: this only works if test_set is frequency-balanced !!!!!!!!
        target_counts[c] = math.ceil(
            desired_class_cardinality / max_desired_class_cardinality * current_class_cardinality
        )

    # Sample indices from each class
    selected_indices = []
    for c, n in target_counts.items():
        sampled = np.random.choice(test_indices_by_class[c], size=n, replace=False).tolist()
        selected_indices.extend(sampled)

    # Build new test set
    images, labels = [], []
    for idx in selected_indices:
        image, label, _ = test_set[idx]
        images.append(image)
        labels.append(label)

    new_images = torch.stack(images)
    new_labels = torch.tensor(labels)
    new_dataset = IndexedDataset(TensorDataset(new_images, new_labels))
    return new_dataset


def get_dataloader(
        dataset: IndexedDataset,
        batch_size: int,
        shuffle: bool = False
):
    """Create a DataLoader with deterministic worker initialization."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, worker_init_fn=worker_init_fn)


def load_real_dataset(
        dataset_name: str,
        shuffle: bool = False,
        apply_augmentation: bool = False,
        synthetic_loader: Optional[DataLoader] = None
) -> Tuple[DataLoader[IndexedDataset], IndexedDataset, DataLoader[IndexedDataset], IndexedDataset]:
    """Load the dataset giving control over shuffling and augmentation. Currently only supports CIFAR-100.

    :param dataset_name: Name of the dataset to load (only accepts `CIFAR-100`).
    :param shuffle: Raise this flag to shuffle the training dataset.
    :param apply_augmentation: Raise this flag to apply data augmentation to the training set.
    :param synthetic_loader: Pass this to adjust the distribution of test set (used for custom FID).

    :return: Tuple containing DataLoader for the training set, training set, DataLoader for the test set, and test set.
    """
    config = get_config(dataset_name)

    train_transform, test_transform = get_transform(apply_augmentation, config)
    if dataset_name == 'CIFAR-100':
        training_set = torchvision.datasets.CIFAR100(root=os.path.join(ROOT, 'data'), download=True,
                                                     transform=train_transform)
        test_set = torchvision.datasets.CIFAR100(root=os.path.join(ROOT, 'data'), train=False, download=True,
                                                 transform=test_transform)
    else:
        raise Exception

    training_set = IndexedDataset(training_set)
    test_set = IndexedDataset(test_set)

    if synthetic_loader is not None:
        test_set = downsample_test_set_to_match_synthetic(test_set, synthetic_loader)

    training_loader = get_dataloader(training_set, config['batch_size'], shuffle)
    test_loader = get_dataloader(test_set, config['batch_size'])

    return training_loader, training_set, test_loader, test_set


def load_synthetic_dataset(
        dataset_name: str,
        generative_model: str,
        normalize: bool
) -> Tuple[DataLoader[IndexedDataset], IndexedDataset]:
    config = get_config(dataset_name)
    unnormalized_transform, normalized_transform = get_transform(False, config)
    path = os.path.join(ROOT, 'synthetic_data', dataset_name, generative_model)

    if dataset_name == 'CIFAR-100':
        if normalize:
            synthetic_set = SyntheticImageDataset(path, transform=normalized_transform)
        else:
            synthetic_set = SyntheticImageDataset(path, transform=unnormalized_transform)
    else:
        raise Exception

    synthetic_set = IndexedDataset(synthetic_set)
    synthetic_loader = get_dataloader(synthetic_set, config['batch_size'] * 2)

    return synthetic_loader, synthetic_set
