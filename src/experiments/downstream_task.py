"""
This module trains an ensemble on the balanced, full-sized dataset and uses it to produce in-hoc hardness estimates.
These estimates will later be used to compute the resampling ratios for our Hardness-Based Resampling.
"""

import argparse
import os
import zipfile

from src.config.config import ROOT
from src.data.loading import load_real_dataset, load_synthetic_dataset
from src.training.train_models import ModelTrainer


def main(dataset_name: str, generative_model: str):
    _, _, test_loader, _ = load_real_dataset(dataset_name)

    syn_path = os.path.join(ROOT, 'synthetic_data', dataset_name, generative_model)
    with zipfile.ZipFile(os.path.join(syn_path, f"{generative_model}.zip"), 'r') as zip_ref:
        zip_ref.extractall(syn_path)
    synthetic_loader, synthetic_set = load_synthetic_dataset(dataset_name, generative_model, False)
    training_set_size = len(synthetic_set)
    training_loaders = [synthetic_loader]
    save_suffix = ''
    trainer = ModelTrainer(training_set_size, training_loaders, test_loader, dataset_name, save_suffix,
                           estimate_hardness=True)
    trainer.train_ensemble()

    # TODO: Modify load_synthetic_dataset so that it works with zipped folders
    # TODO: Make the code so that it trains the models on every fifth checkpoint (sorted by epochs)
    # TODO: Ensure that ModelTrainer correctly saves hardness estimates (different folders then the ones on real data)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an ensemble of models on CIFAR-100.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['CIFAR-100'], help='Dataset name: CIFAR-100')
    parser.add_argument('--generative_model', type=str, required=True, choices=['edm', 'stylegan'],
                        help='Name of the generative model architecture used to produce the synthetic data.')

    args = parser.parse_args()
    main(args.dataset_name, args.generative_model)
