"""
This module loads the models pre-trained on balanced dataset to measure and save the post-hoc hardness.
"""

import argparse
import os

from src.config.config import ROOT
from src.data.loading import load_real_dataset, load_synthetic_dataset
from src.measures.inception_score import compute_inception_score
from src.measures.frechet_inception_distance import compute_fid


def main(dataset_name: str):
    synth_root = os.path.join(ROOT, 'synthetic_data', dataset_name)
    generative_models = sorted([d for d in os.listdir(synth_root) if os.path.isdir(os.path.join(synth_root, d))])
    _, _, test_loader, _ = load_real_dataset(dataset_name)

    for generative_model in generative_models:
        for model_type in ['ResNet18LowRes', 'InceptionV3']:
            synthetic_loader, _ = load_synthetic_dataset(dataset_name, generative_model, False)
            # IS_mean, IS_std = compute_inception_score(synthetic_loader, dataset_name, model_type)
            # print(f'IS for {generative_model} using {model_type} (dataset norm) = {IS_mean:.2f} ± {IS_std:.2f}')
            fid = compute_fid(dataset_name, test_loader, synthetic_loader, model_type)
            print(f'FID for {generative_model} using {model_type} as feature extractor equals to {fid}.')
            if model_type == 'InceptionV3':
                # IS_mean, IS_std = compute_inception_score(synthetic_loader, dataset_name, model_type, True)
                # print(f'IS for {generative_model} using {model_type} (ImageNet norm) = '
                #       f'{is_mean_im:.4f} ± {is_std_im:.4f}')
                fid = compute_fid(dataset_name, test_loader, synthetic_loader, model_type, True)
                print(f"FID for {generative_model} using {model_type} as feature extractor and normalising with "
                      f"ImageNet's mean and std equals to {fid}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an ensemble of models on CIFAR-100.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['CIFAR-100'], help='Dataset name: CIFAR-100')

    args = parser.parse_args()
    main(args.dataset_name)
