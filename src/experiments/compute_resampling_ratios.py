"""
This module takes the in_hoc_hardness_estimates.pkl obtained by running src/experiments/estimate_hardness_in_hoc.py and
computes the resampling ratios for the downstream task of hardness-based resampling.
"""

import argparse
import json
import os
import pickle
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from src.config.config import ROOT, get_config
from src.data.loading import load_real_dataset
from src.utils.evaluation import compute_sample_allocation_for_resampling
from src.utils.io import load_in_hoc_hardness_estimates
from src.utils.reproducibility import set_reproducibility


def plot_resampling_counts_stability(all_resampling_counts: List[List[int]], num_classes: int):
    plt.figure(figsize=(10, 6))
    for idx, resampling_counts in enumerate(all_resampling_counts):
        plt.plot(range(num_classes), resampling_counts, alpha=0.5, label=f'Estimate {idx+1}')
    plt.xlabel('Class index')
    plt.ylabel('Class cardinalities for resampling')
    plt.title('Individual class cardinalities for resampling')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig('resampling_counts_stability.pdf', dpi=150)


def visualize_resampling_results(
        sample_allocation: List[int],
        num_classes: int,
        dataset_name: str
):
    """Produces components of Figure 4.

    This visualization shows the effects of resampling. It clearly shows how many samples were removed from each
    class due to hardness-based resampling (the classes are sorted for clarity). It also computes the number of
    easy classes in the dataset - the classes that are undersampled during hardness-based resampling."""
    avg_count = np.mean(sample_allocation)
    min_count = min(sample_allocation)
    max_count = max(sample_allocation)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhspan(min_count, avg_count, color='green', alpha=0.15)
    ax.axhline(y=avg_count, color='black', linestyle='--', linewidth=2)
    ax.axhspan(avg_count, max_count, color='red', alpha=0.15)

    sorted_indices = np.argsort(sample_allocation)
    sorted_counts = np.array(sample_allocation)[sorted_indices]
    ax.plot(range(len(sorted_counts)), sorted_counts, color='grey', linewidth=2)

    ax.set_xlabel('Classes sorted based on hardness (hardest to the right)')
    ax.set_xticklabels([])
    ax.set_xticks(np.arange(0, num_classes + 1))
    ax.set_ylabel('Class-wise sample count after resampling')
    ax.set_title(dataset_name)
    figure_save_dir = os.path.join(ROOT, 'Figures/', dataset_name)
    fig.savefig(os.path.join(figure_save_dir, 'sorted_resampled_dataset.pdf'))

    number_of_easy_classes = sum([sample_allocation[cls] <= avg_count for cls in range(len(sample_allocation))])
    print(f'Identified {number_of_easy_classes} easy classes in this dataset.')


def main(dataset_name: str):
    config = get_config(dataset_name)
    num_classes = config['num_classes']
    num_training_samples = config['num_training_samples']

    training_loader, training_set, _, _ = load_real_dataset(dataset_name)
    labels = []
    for i in range(len(training_set)):
        labels.append(training_set[i][1])
    labels = np.array(labels)

    in_hoc_hardness_estimates = load_in_hoc_hardness_estimates(dataset_name)
    set_reproducibility()

    resampling_counts = []
    for idx in range(9):
        avg_hardness = list(np.mean([in_hoc_hardness_estimates[idx]], axis=0))
        resampling_counts.append(compute_sample_allocation_for_resampling(avg_hardness, labels, num_classes,
                                                                          sum(num_training_samples), 2.5))
    plot_resampling_counts_stability(resampling_counts, num_classes)
    visualize_resampling_results(resampling_counts[0], num_classes, dataset_name)

    oversampling_targets = [max(0, resampling_counts[0][cls_idx] - num_training_samples[cls_idx])
                            for cls_idx in range(num_classes)]
    with open('oversampling_targets.json', 'w') as f:
        json.dump(oversampling_targets, f)
    hardness_save_dir = os.path.join(ROOT, "Results", dataset_name, f'alpha_2.00')
    os.makedirs(hardness_save_dir, exist_ok=True)
    with open(os.path.join(hardness_save_dir, 'samples_per_class.pkl'), 'wb') as file:
        pickle.dump(resampling_counts[0], file)

    print(oversampling_targets)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute resampling ratios for Hardness Based Resampling.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['CIFAR-100'], help='Dataset name: CIFAR-100')

    args = parser.parse_args()
    main(args.dataset_name)
