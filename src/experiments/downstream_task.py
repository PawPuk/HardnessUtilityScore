"""
This module trains an ensemble on the balanced, full-sized dataset and uses it to produce in-hoc hardness estimates.
These estimates will later be used to compute the resampling ratios for our Hardness-Based Resampling.
"""

import argparse
import os
import shutil
import zipfile

from src.config.config import ROOT
from src.data.loading import load_real_dataset, load_synthetic_dataset
from src.training.train_models import ModelTrainer


def main(dataset_name: str, generative_architecture: str):
    _, _, test_loader, _ = load_real_dataset(dataset_name)

    synth_root = os.path.join(ROOT, 'synthetic_data', dataset_name)
    generative_model_zips = sorted(
        f for f in os.listdir(synth_root)
        if os.path.isfile(os.path.join(synth_root, f)) and 'metrics' not in f and generative_architecture in f
    )

    for zip_file in generative_model_zips:
        generative_model = os.path.splitext(zip_file)[0]
        with zipfile.ZipFile(os.path.join(synth_root, f"{generative_model}.zip"), 'r') as zip_ref:
            zip_ref.extractall(synth_root)

        synthetic_loader, synthetic_set = load_synthetic_dataset(dataset_name, generative_model, False)
        training_set_size = len(synthetic_set)
        training_loaders = [synthetic_loader]
        save_suffix = ''
        trainer = ModelTrainer(training_set_size, training_loaders, test_loader, dataset_name, save_suffix,
                               estimate_hardness=True)
        trainer.train_ensemble()

        shutil.rmtree(os.path.join(synth_root, generative_model))

    # TODO: Make the code so that it trains the models on every fifth checkpoint (sorted by epochs)
    # TODO: Ensure that ModelTrainer correctly saves hardness estimates (different folders then the ones on real data)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an ensemble of models on CIFAR-100.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['CIFAR-100'], help='Dataset name: CIFAR-100')
    parser.add_argument('--generative_architecture', type=str, required=True, choices=['edm', 'stylegan'],
                        help='Name of the architecture of the generative model used to produce the synthetic data.')

    args = parser.parse_args()
    main(args.dataset_name, args.generative_architecture)
