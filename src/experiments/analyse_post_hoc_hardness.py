import argparse
from glob import glob
import os
import pickle
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from src.config.config import ROOT


def compute_avg_hardness_per_model(pkl_path: str) -> np.ndarray:
    """
    Load a pickle file (dict: model_idx -> list of floats),
    average over model indices -> 1D array (number of samples).
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)  # dict[int, List[float]]
    sorted_indices = sorted(data.keys())  # Convert to 2D array (models x samples)
    arr = np.array([data[idx] for idx in sorted_indices])
    avg_over_models = np.mean(arr, axis=0)
    return avg_over_models


def plot_hardness_histograms_for_type(
        real_hardness: np.ndarray,
        model_files: List[str],
        hardness_type: str
) -> None:
    """
    For a given hardness type (e.g., 'logit_margins'), find all model .pkl files
    that contain that type in their name (excluding real_test files), average over
    model indices, and plot a histogram for each model. If a real_test file
    named `real_test_{hardness_type}.pkl` exists, overlay its hardness distribution
    as a black line in every subplot.
    """
    n_files = len(model_files)
    cols = min(4, n_files)
    rows = (n_files + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 10))
    axes = axes.flatten() if n_files > 1 else [axes]

    # For each model file, compute averaged hardness and plot histogram
    for ax, fpath in zip(axes, model_files):
        hardness = compute_avg_hardness_per_model(fpath)

        ax.hist(hardness, bins=50, alpha=0.7, edgecolor='black', density=True, label='Model')
        ax.hist(real_hardness, bins=50, histtype='step', color='black', linewidth=2, density=True, label='Real')

        ax.set_title(os.path.basename(fpath).replace('.pkl', ''))
        ax.set_xlabel('Hardness')
        ax.set_ylabel('Density')

    # Hide any unused subplots
    for ax in axes[n_files:]:
        ax.axis('off')

    plt.suptitle(f"Hardness type: {hardness_type}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
    plt.savefig(f"hardness_histograms_{hardness_type}.png", dpi=150, bbox_inches='tight')
    plt.show()


def main(dataset_name: str):
    base_dir = os.path.join(ROOT, 'Results', dataset_name, "post_hoc_hardness_estimates")
    hardness_types = ["logit_margins"]

    all_files = glob(os.path.join(base_dir, "*.pkl"))
    if not all_files:
        raise Exception(f"No .pkl files found in {base_dir}")

    for hardness_type in hardness_types:
        # Separate the real_test file for this type and model files
        real_file, model_files = None, []
        for f in all_files:
            basename = os.path.basename(f)
            if hardness_type in basename:
                if basename == f"real_test_{hardness_type}.pkl":
                    real_file = f
                else:
                    model_files.append(f)
        real_hardness = compute_avg_hardness_per_model(real_file)

        plot_hardness_histograms_for_type(real_hardness, sorted(model_files), hardness_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an ensemble of models on CIFAR-100.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['CIFAR-100'], help='Dataset name: CIFAR-100')

    args = parser.parse_args()
    main(args.dataset_name)
