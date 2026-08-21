"""
This module loads the models pre-trained on balanced dataset to measure and save the post-hoc hardness.
"""

import argparse
import os
import pickle
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
        num_classes: int,
        use_logits: bool
) -> Tuple[Dict[int, List[float]], Dict[int, List[float]]]:
    confidences, margins = {}, {}
    for model_idx in tqdm(model_paths[0].keys(), desc='Iterating through model indices'):
        model = ResNet18LowRes(num_classes=num_classes).to(DEVICE)
        model.load_state_dict(torch.load(model_paths[0][model_idx]))
        model.eval()
        confidences[model_idx], margins[model_idx] = compute_margins_and_confidences(model, loader, use_logits)
    return confidences, margins


def main(
        dataset_name: str,
        overwrite: bool,
        use_logits: bool
):
    config = get_config(dataset_name)
    num_classes = config['num_classes']

    synth_root = os.path.join(ROOT, 'synthetic_data', dataset_name)
    generative_models = sorted([d for d in os.listdir(synth_root) if os.path.isdir(os.path.join(synth_root, d))])

    model_paths = extract_paths_to_pretrained_models(dataset_name)
    save_dir = os.path.join(ROOT, "Results", dataset_name, 'post_hoc_hardness_estimates')
    os.makedirs(save_dir, exist_ok=True)
    suffix = 'logit' if use_logits else 'softmax'

    for generative_model in tqdm(generative_models):
        save_path_confidences = os.path.join(save_dir, f"{generative_model}_{suffix}_confidences.pkl")
        save_path_margins = os.path.join(save_dir, f"{generative_model}_{suffix}_margins.pkl")
        if os.path.exists(save_path_confidences) and os.path.exists(save_path_margins) and not overwrite:
            continue
        print(f'Estimating post-hoc hardness for {generative_model}.')

        synthetic_loader, _ = load_synthetic_dataset(dataset_name, generative_model, True)
        confidences, margins = estimate_and_save_post_hoc_hardness(
            synthetic_loader, model_paths, num_classes, use_logits
        )
        for path, hardness_estimates in [(save_path_confidences, confidences), (save_path_margins, margins)]:
            with open(path, "wb") as file:
                pickle.dump(hardness_estimates, file)

    save_path_confidences = os.path.join(save_dir, f'real_test_{suffix}_confidences.pkl')
    save_path_margins = os.path.join(save_dir, f'real_test_{suffix}_margins.pkl')
    if not (os.path.exists(save_path_confidences) and os.path.exists(save_path_margins)) or overwrite:
        print('Estimating post-hoc hardness for real data.')

        _, _, test_loader, _ = load_real_dataset(dataset_name)
        confidences, margins = estimate_and_save_post_hoc_hardness(test_loader, model_paths, num_classes, use_logits)
        for path, hardness_estimates in [(save_path_confidences, confidences), (save_path_margins, margins)]:
            with open(path, "wb") as file:
                pickle.dump(hardness_estimates, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an ensemble of models on CIFAR-100.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['CIFAR-100'], help='Dataset name: CIFAR-100')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite existing margin files if they already exist.')
    parser.add_argument('--use_logits', action='store_true', default=False,
                        help='Raise this flag to use logits. Otherwise softmax probabilities will be computed')

    args = parser.parse_args()
    main(args.dataset_name, args.overwrite, args.use_logits)
