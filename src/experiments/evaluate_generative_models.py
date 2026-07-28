"""
This module loads the models pre-trained on balanced dataset to measure and save the post-hoc hardness.
"""

import argparse
import os

from src.config.config import ROOT
from src.data.loading import load_synthetic_dataset
from src.measures.inception_score import compute_inception_score_using_inceptionV3


def main(dataset_name: str):
    synth_root = os.path.join(ROOT, 'synthetic_data', dataset_name)
    generative_models = sorted([d for d in os.listdir(synth_root) if os.path.isdir(os.path.join(synth_root, d))])

    for generative_model in generative_models:
        synthetic_loader, _ = load_synthetic_dataset(dataset_name, generative_model)
        inception_score = compute_inception_score_using_inceptionV3(synthetic_loader)
        print(inception_score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an ensemble of models on CIFAR-100.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['CIFAR-100'], help='Dataset name: CIFAR-100')

    args = parser.parse_args()
    main(args.dataset_name)
