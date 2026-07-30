"""
This module loads the models pre-trained on balanced dataset to measure and save the post-hoc hardness.
"""

import argparse
import os
import pickle

import torch
from tqdm import tqdm

from src.config.config import DEVICE, ROOT, get_config
from src.data.loading import load_synthetic_dataset
from src.measures.hardness_estimators import compute_margins
from src.models.neural_networks import ResNet18LowRes
from src.utils.io import extract_paths_to_pretrained_models


def main(dataset_name: str):
    config = get_config(dataset_name)
    num_classes = config['num_classes']

    synth_root = os.path.join(ROOT, 'synthetic_data', dataset_name)
    generative_models = sorted([d for d in os.listdir(synth_root) if os.path.isdir(os.path.join(synth_root, d))])

    model_paths = extract_paths_to_pretrained_models(dataset_name)

    # For each generative model, compute margins and save
    for generative_model in generative_models:
        # Load the corresponding synthetic data
        synthetic_loader, _ = load_synthetic_dataset(dataset_name, generative_model, True)

        # Compute margins
        margins = {}
        for model_idx in tqdm(model_paths[0].keys(), desc='Iterating through model indices'):
            model = ResNet18LowRes(num_classes=num_classes).to(DEVICE)
            model.load_state_dict(torch.load(model_paths[0][model_idx]))
            model.eval()
            margins[model_idx] = compute_margins(model, synthetic_loader)

        # Save margins
        save_dir = os.path.join(ROOT, "Results", dataset_name, 'post_hoc_hardness_estimates')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{generative_model}_margins.pkl")
        with open(save_path, "wb") as file:
            pickle.dump(margins, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an ensemble of models on CIFAR-100.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['CIFAR-100'], help='Dataset name: CIFAR-100')

    args = parser.parse_args()
    main(args.dataset_name)
