"""
This module trains an ensemble on the balanced, full-sized dataset and uses it to produce in-hoc hardness estimates.
These estimates will later be used to compute the resampling ratios for our Hardness-Based Resampling.
"""

import argparse

import numpy as np

from src.config.config import get_config
from src.data.loading import get_dataloader, load_real_dataset, load_synthetic_dataset, perform_data_augmentation
from src.resampling.resampling import DataResampling
from src.training.train_models import ModelTrainer
from src.utils.evaluation import compute_sample_allocation_for_resampling
from src.utils.io import load_in_hoc_hardness_estimates, load_post_hoc_hardness_estimates
from src.utils.reproducibility import set_reproducibility


def main(dataset_name: str, generative_model: str, oversampling_strategy: str, alpha: float):
    config = get_config(dataset_name)
    num_classes = config['num_classes']
    num_training_samples = config['num_training_samples']
    dataset_count = config['num_datasets']

    _, training_set, test_loader, _ = load_real_dataset(dataset_name)
    _, synthetic_set = load_synthetic_dataset(dataset_name, generative_model, False)

    labels = [training_set[idx][1] for idx in range(len(training_set))]
    # Not doing np.mean() here to match the in_hoc_estimates obtained from compute_resampling_ratios.py
    in_hoc_hardness_estimates = load_in_hoc_hardness_estimates(dataset_name)

    samples_per_class = compute_sample_allocation_for_resampling(in_hoc_hardness_estimates[0], labels, num_classes,
                                                                 sum(num_training_samples), alpha=alpha)

    in_hoc_hardness_estimates = np.mean(in_hoc_hardness_estimates, axis=0)
    post_hoc_hardness_estimates = np.mean(load_post_hoc_hardness_estimates(dataset_name, generative_model), axis=0)

    resampled_loaders = []
    for dataset_idx in range(dataset_count):
        set_reproducibility(42 * dataset_idx)
        resampler = DataResampling(training_set, num_classes, oversampling_strategy, in_hoc_hardness_estimates,
                                   post_hoc_hardness_estimates, synthetic_set)
        resampled_dataset = resampler.resample_data(samples_per_class)
        print(f'Resampled dataset contains {len(resampled_dataset)} data samples.')

        augmented_resampled_dataset = perform_data_augmentation(resampled_dataset, dataset_name)
        resampled_loaders.append(get_dataloader(augmented_resampled_dataset, batch_size=config['batch_size'],
                                                shuffle=True))

    save_suffix = f'{oversampling_strategy}_{generative_model}_{alpha:.2f}'
    trainer = ModelTrainer(len(training_set), resampled_loaders, test_loader, dataset_name, save_suffix,
                           save_probe_models=False)

    trainer.train_ensemble()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an ensemble of models on CIFAR-100.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['CIFAR-100'], help='Dataset name: CIFAR-100')
    parser.add_argument('--generative_model', type=str, required=True, choices=['edm', 'vae'],
                        help='Name of the generative model architecture used to produce the synthetic data.')
    parser.add_argument('--oversampling_strategy', type=str, required=True, choices=['random', 'hard'],
                        help='Pick `random` to use random synthetic samples. Pick `hard` to use only the hardest ones '
                             'for oversampling.')
    parser.add_argument('--alpha', type=float, default=1.00, help='Used to control the degree of introduced imbalance.')

    args = parser.parse_args()
    main(args.dataset_name, args.generative_model, args.oversampling_strategy, args.alpha)
