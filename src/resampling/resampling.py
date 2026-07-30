"""Core module for hardness-based resampling that returns the resampled dataset."""

import random
from typing import List

import numpy as np
import torch
from torch.utils.data import TensorDataset

from src.data.datasets import IndexedDataset


class DataResampling:
    """Class that contains all the methods required for hardness-based resampling"""
    def __init__(self, dataset: IndexedDataset, num_classes: int, oversampling_strategy: str,
                 in_hoc_hardness_estimates: List[float], post_hoc_hardness_estimates: List[float],
                 synthetic_set: IndexedDataset):
        """Initialize the DataResampling class.

        :param dataset: The hardness-based resampling will be applied to this dataset
        :param num_classes: Number of classes in the dataset
        :param oversampling_strategy: Name of the oversampling strategy
        :param in_hoc_hardness_estimates: In-hoc hardness estimates for the real dataset
        :param post_hoc_hardness_estimates: Post-hoc hardness estimates for the synthetic data
        :param synthetic_set: Contains the synthetic data samples for oversampling

        """
        self.dataset = dataset
        self.num_classes = num_classes
        self.oversampling_strategy = oversampling_strategy
        self.in_hoc_hardness_estimates = in_hoc_hardness_estimates
        self.post_hoc_hardness_estimates = post_hoc_hardness_estimates
        self.synthetic_set = synthetic_set

    @staticmethod
    def prune_easy(desired_count: int, hardness_scores: List[float]) -> List[int]:
        """Prune based on hardness focusing on the removal of easy samples.
        :returns: indices of the samples to keep after pruning"""
        sorted_indices = np.argsort(hardness_scores)
        return list(sorted_indices[:desired_count])

    def holdout_oversample(self, desired_count: int, class_id: int, hardness_scores: List[float]) -> torch.Tensor:
        """Perform random oversampling to match the desired count using the holdout set."""

        synthetic_images = [self.synthetic_set[idx][0] for idx in range(len(self.synthetic_set))
                            if self.synthetic_set[idx][1] == class_id]
        if len(synthetic_images) < desired_count:
            raise ValueError(f"Holdout set only has {len(synthetic_images)} samples for class {class_id}, need "
                             f"{desired_count}")

        if self.oversampling_strategy == 'random':
            selected_indices = random.sample(range(len(synthetic_images)), desired_count)
        else:
            sorted_indices = np.argsort(hardness_scores)
            selected_indices = list(sorted_indices[:desired_count])

        synthetic_images = torch.stack([synthetic_images[i] for i in selected_indices])

        return synthetic_images

    def resample_data(self, samples_per_class: List[int]):
        """
        Perform resampling to match the desired samples_per_class. Uses the selected undersampling and oversampling
        methods.
        ------------------------------------------------------------------------------------------------
        There are two types of oversampling strategies - ones that generate new data and one that reuse existing data.
        """

        # Organize labels and hardness estimates by classes
        data_by_class = {i: [] for i in range(self.num_classes)}
        in_hoc_hardness_by_class = {i: [] for i in range(self.num_classes)}
        post_hoc_hardness_by_class = {i: [] for i in range(self.num_classes)}
        for image, label, idx in self.dataset:
            data_by_class[label].append(image)
            in_hoc_hardness_by_class[label].append(self.in_hoc_hardness_estimates[idx])
        for image, label, idx in self.synthetic_set:
            post_hoc_hardness_by_class[label].append(self.post_hoc_hardness_estimates[idx])

        resampled_data, resampled_labels = [], []

        # Perform resampling for each class
        for class_id, hardnesses_within_class in in_hoc_hardness_by_class.items():
            current_images = data_by_class[class_id]
            current_post_hoc_hardness = post_hoc_hardness_by_class[class_id]
            current_count = len(current_images)
            desired_count = samples_per_class[class_id]

            if current_count > desired_count:
                class_retain_indices = self.prune_easy(desired_count, hardnesses_within_class)
                kept_images = [current_images[pos] for pos in class_retain_indices]
                resampled_data.extend(kept_images)
                resampled_labels.extend([class_id] * len(kept_images))
            elif current_count < desired_count:
                n_extra = desired_count - current_count
                synthetic_imgs = self.holdout_oversample(n_extra, class_id, current_post_hoc_hardness)

                resampled_data.extend(current_images)
                resampled_data.extend(synthetic_imgs)
                resampled_labels.extend([class_id] * (current_count + n_extra))
            elif current_count == desired_count:
                resampled_data.extend(current_images)
                resampled_labels.extend([class_id] * current_count)

        new_images = torch.stack(resampled_data)
        new_labels = torch.tensor(resampled_labels)
        new_tensor_dataset = TensorDataset(new_images, new_labels)

        return IndexedDataset(new_tensor_dataset)
