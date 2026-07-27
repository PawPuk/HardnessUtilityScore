"""
This module takes the in_hoc_hardness_estimates.pkl obtained by running src/experiments/estimate_hardness_in_hoc.py and
computes the resampling ratios for the downstream task of hardness-based resampling.
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from src.config.config import get_config
from src.data.loading import load_real_dataset
from src.utils.evaluation import compute_sample_allocation_for_resampling
from src.utils.io import load_in_hoc_hardness_estimates
from src.utils.reproducibility import set_reproducibility


def plot_individual_hardness(all_class_level_AUMs, num_classes):
    plt.figure(figsize=(10, 6))
    for idx, class_means in enumerate(all_class_level_AUMs):
        plt.plot(range(num_classes), class_means, alpha=0.5, label=f'Estimate {idx+1}')
    plt.xlabel('Class index')
    plt.ylabel('Class cardinalities for resampling')
    plt.title('Individual class cardinalities for resampling')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig('resampling_counts_stability.pdf', dpi=150)


def main(dataset_name: str):
    config = get_config(dataset_name)
    num_classes = config['num_classes']
    num_training_samples = config['num_training_samples']

    training_loader, training_set, _, _ = load_real_dataset(dataset_name)
    labels = []
    for i in range(len(training_set)):
        labels.append(training_set[i][1].item())
    labels = np.array(labels)

    in_hoc_hardness_estimates = load_in_hoc_hardness_estimates(dataset_name)
    set_reproducibility()

    resampling_counts = []
    for idx in range(9):
        avg_hardness = list(np.mean([in_hoc_hardness_estimates[idx]], axis=0))
        resampling_counts.append(compute_sample_allocation_for_resampling(avg_hardness, labels, num_classes,
                                                                          sum(num_training_samples), 2.5))
    plot_individual_hardness(resampling_counts, num_classes)
    oversampling_targets = [max(0, resampling_counts[0][cls_idx] - num_training_samples[cls_idx])
                            for cls_idx in range(num_classes)]
    with open('oversampling_targets.json', 'w') as f:
        json.dump(oversampling_targets, f)
    print(oversampling_targets)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute resampling ratios for Hardness Based Resampling.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['CIFAR-100'], help='Dataset name: CIFAR-100')

    args = parser.parse_args()
    main(args.dataset_name)
