"""
This module takes the in_hoc_hardness_estimates.pkl obtained by running src/experiments/estimate_hardness_in_hoc.py and
performs the stability analysis to identify how the class-level estimates vary based on the model initialization (which
affected AUM).
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from src.config.config import get_config
from src.data.loading import load_real_dataset
from src.utils.io import load_in_hoc_hardness_estimates
from src.utils.reproducibility import set_reproducibility


def plot_individual_hardness(all_class_level_AUMs, num_classes):
    plt.figure(figsize=(10, 6))
    for idx, class_means in enumerate(all_class_level_AUMs):
        plt.plot(range(num_classes), class_means, alpha=0.5, label=f'Estimate {idx+1}')
    plt.xlabel('Class index')
    plt.ylabel('Class-level in-hoc hardness (AUM)')
    plt.title('Individual class-level hardness estimates from 9 runs')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig('class_level_AUM.pdf', dpi=150)


def main(dataset_name: str):
    config = get_config(dataset_name)
    num_classes = config['num_classes']
    num_training_samples = config['num_training_samples']

    training_loader, training_set, _, _ = load_real_dataset(dataset_name)
    labels = []
    for i in range(len(training_set)):
        labels.append(training_set[i][1].item())
    assert [labels.count(i) for i in range(num_classes)] == num_training_samples  # Sanity check.
    labels = np.array(labels)

    in_hoc_hardness_estimates = load_in_hoc_hardness_estimates(dataset_name)
    set_reproducibility()

    all_class_level_AUMs = []
    for idx in range(9):
        hardness_estimates_by_class = [[] for _ in range(num_classes)]
        for sample_idx, label in enumerate(labels):
            hardness_estimates_by_class[label].append(in_hoc_hardness_estimates[idx][sample_idx])
        class_means = [np.mean(h_list) for h_list in hardness_estimates_by_class]
        all_class_level_AUMs.append(class_means)

    plot_individual_hardness(all_class_level_AUMs, num_classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyse the in-hoc hardness estimates (specifically, AUM).')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['CIFAR-100'], help='Dataset name: CIFAR-100')

    args = parser.parse_args()
    main(args.dataset_name)
