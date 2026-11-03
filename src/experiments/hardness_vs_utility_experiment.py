"""
This module trains an ensemble on the balanced, full-sized dataset and uses it to produce in-hoc hardness estimates.
These estimates will later be used to compute the resampling ratios for our Hardness-Based Resampling.
"""

import argparse
import os
import pickle
import shutil
import zipfile

import numpy as np
from torch.utils.data import Subset

from src.config.config import ROOT, get_config
from src.data.loading import get_dataloader, load_real_dataset, load_synthetic_dataset, IndexedDataset
from src.training.train_models import ModelTrainer
from src.utils.structures import load_labels


def main(dataset_name: str, generative_architecture: str):
    _, _, test_loader, _ = load_real_dataset(dataset_name)

    synth_root = os.path.join(ROOT, 'synthetic_data', dataset_name)
    generative_model_zips = sorted(
        f for f in os.listdir(synth_root)
        if os.path.isfile(os.path.join(synth_root, f)) and 'metrics' not in f and generative_architecture in f
    )
    for i, zip_file in enumerate(generative_model_zips):
        if generative_architecture == 'edm' and i not in [4, 14]:
            continue
        if generative_architecture == 'stylegan' and i not in [4, 14]:
            continue
        generative_model = os.path.splitext(zip_file)[0]
        with zipfile.ZipFile(os.path.join(synth_root, f"{generative_model}.zip"), 'r') as zip_ref:
            zip_ref.extractall(synth_root)

        _, synthetic_set = load_synthetic_dataset(dataset_name, generative_model, True)
        hardness_estimates_file = os.path.join(ROOT, 'Results', dataset_name, 'post_hoc_hardness_estimates',
                                               f'{generative_model}_logit_margins.pkl')
        with open(hardness_estimates_file, 'rb') as f:
            data = pickle.load(f)
        sorted_indices = sorted(data.keys())
        arr = np.array([data[idx] for idx in sorted_indices])
        avg_over_models = np.mean(arr, axis=0)

        synthetic_labels = load_labels(synthetic_set)
        selected_global_indices = {i1: [] for i1 in range(4)}
        for class_idx in range(get_config(dataset_name)['num_classes']):
            mask = (synthetic_labels == class_idx)
            class_global_indices = np.where(mask)[0]
            hardness_by_class = avg_over_models[mask]

            for strat_i in range(4):
                stratification = np.argsort(hardness_by_class)[strat_i*100:strat_i*100+200]
                selected_global_indices[strat_i].extend(class_global_indices[stratification].tolist())
        selected_global_indices = {k: np.array(l) for k, l in selected_global_indices.items()}

        for strat_i in range(len(selected_global_indices.keys())):
            training_set_size = len(selected_global_indices[strat_i])
            stratified_subdataset = Subset(synthetic_set, selected_global_indices[strat_i])
            stratified_loader = get_dataloader(stratified_subdataset, get_config(dataset_name)['batch_size'], True)
            training_loaders = [stratified_loader]
            save_suffix = f'{generative_model}_strat{strat_i}'
            trainer = ModelTrainer(training_set_size, training_loaders, test_loader, dataset_name, save_suffix,
                                   save_probe_models=False)
            trainer.train_ensemble()

        shutil.rmtree(os.path.join(synth_root, generative_model))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an ensemble of models on CIFAR-100.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['CIFAR-100'], help='Dataset name: CIFAR-100')
    parser.add_argument('--generative_architecture', type=str, required=True, choices=['edm', 'stylegan'],
                        help='Name of the architecture of the generative model used to produce the synthetic data.')

    args = parser.parse_args()
    main(args.dataset_name, args.generative_architecture)
