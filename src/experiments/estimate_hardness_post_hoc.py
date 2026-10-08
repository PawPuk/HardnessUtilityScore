"""
This module loads the models pre-trained on balanced dataset to measure and save the post-hoc hardness.
"""

import argparse
import os
import pickle
import shutil
import zipfile
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from src.config.config import DEVICE, ROOT, get_config
from src.data.loading import load_synthetic_dataset, load_real_dataset
from src.measures.hardness_estimators import compute_margins_and_confidences
from src.models.neural_networks import ResNet18LowRes
from src.utils.io import extract_paths_to_pretrained_models


def estimate_and_save_post_hoc_hardness(
        loader: torch.utils.data.DataLoader,
        model_paths: Dict[int, Dict[int, str]],
        num_classes: int
) -> Tuple[Dict[int, List[float]], Dict[int, List[float]], Dict[int, List[float]], Dict[int, List[float]]]:
    logit_confidences, softmax_confidences, logit_margins, softmax_margins = {}, {}, {}, {}
    for i in tqdm(model_paths[0].keys(), desc='Iterating through model indices'):
        model = ResNet18LowRes(num_classes=num_classes).to(DEVICE)
        model.load_state_dict(torch.load(model_paths[0][i]))
        model.eval()
        logit_confidences[i], softmax_confidences[i], logit_margins[i], softmax_margins[i] = \
            compute_margins_and_confidences(model, loader)
    return logit_confidences, softmax_confidences, logit_margins, softmax_margins


def main(
        dataset_name: str,
        overwrite: bool
):
    config = get_config(dataset_name)
    num_classes = config['num_classes']

    synth_root = os.path.join(ROOT, 'synthetic_data', dataset_name)
    generative_model_zips = sorted(
        f for f in os.listdir(synth_root)
        if os.path.isfile(os.path.join(synth_root, f)) and 'metrics' not in f
    )
    model_paths = extract_paths_to_pretrained_models(dataset_name)
    save_dir = os.path.join(ROOT, "Results", dataset_name, 'post_hoc_hardness_estimates')
    os.makedirs(save_dir, exist_ok=True)

    # Estimate post-hoc hardness for synthetic data
    for zip_file in tqdm(generative_model_zips):
        generative_model = os.path.splitext(zip_file)[0]
        if os.path.exists(os.path.join(save_dir, f"{generative_model}_logit_confidences.pkl")) and not overwrite:
            continue  # Post-hoc hardness have been estimated for this generative_model, so we skip it

        print(f'Extracting {generative_model}.')
        with zipfile.ZipFile(os.path.join(synth_root, f"{generative_model}.zip"), 'r') as zip_ref:
            zip_ref.extractall(synth_root)

        print(f'Estimating post-hoc hardness for {generative_model}.')
        synthetic_loader, _ = load_synthetic_dataset(dataset_name, generative_model, True)
        logit_confidences, softmax_confidences, logit_margins, softmax_margins = estimate_and_save_post_hoc_hardness(
            synthetic_loader, model_paths, num_classes
        )
        with open(os.path.join(save_dir, f"{generative_model}_logit_confidences.pkl"), 'wb') as file:
            pickle.dump(logit_confidences, file)
        with open(os.path.join(save_dir, f"{generative_model}_softmax_confidences.pkl"), 'wb') as file:
            pickle.dump(softmax_confidences, file)
        with open(os.path.join(save_dir, f"{generative_model}_logit_margins.pkl"), 'wb') as file:
            pickle.dump(logit_margins, file)
        with open(os.path.join(save_dir, f"{generative_model}_softmax_margins.pkl"), 'wb') as file:
            pickle.dump(softmax_margins, file)

        shutil.rmtree(os.path.join(synth_root, generative_model))

    # Estimate post-hoc hardness for real test data (for comparison)
    if not (os.path.join(save_dir, f'real_test_logit_confidences.pkl')) or overwrite:
        print('Estimating post-hoc hardness for real data.')

        _, _, test_loader, _ = load_real_dataset(dataset_name)
        logit_confidences, softmax_confidences, logit_margins, softmax_margins = estimate_and_save_post_hoc_hardness(
            test_loader, model_paths, num_classes
        )

        with open(os.path.join(save_dir, f'real_test_logit_confidences.pkl'), 'wb') as file:
            pickle.dump(logit_confidences, file)
        with open(os.path.join(save_dir, f'real_test_softmax_confidences.pkl'), 'wb') as file:
            pickle.dump(softmax_confidences, file)
        with open(os.path.join(save_dir, f'real_test_logit_margins.pkl'), 'wb') as file:
            pickle.dump(logit_margins, file)
        with open(os.path.join(save_dir, f'real_test_softmax_margins.pkl'), 'wb') as file:
            pickle.dump(softmax_margins, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an ensemble of models on CIFAR-100.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['CIFAR-100'], help='Dataset name: CIFAR-100')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite existing margin files if they already exist.')

    args = parser.parse_args()
    main(args.dataset_name, args.overwrite)
