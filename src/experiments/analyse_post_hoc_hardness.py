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


def plot_hardness_for_type(
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

    fig_hist, axes_hist = plt.subplots(rows, cols, figsize=(16, 10))
    fig_sorted, axes_sorted = plt.subplots(rows, cols, figsize=(16, 10))
    axes_hist = axes_hist.flatten()
    axes_sorted = axes_sorted.flatten()

    sorted_real = np.sort(real_hardness)

    # For each model file, compute averaged hardness and plot histogram
    for ax_hist, ax_sorted, fpath in zip(axes_hist, axes_sorted, model_files):
        hardness = compute_avg_hardness_per_model(fpath)

        # ---- Histogram (density) ----
        ax_hist.hist(hardness, bins=50, alpha=0.7, edgecolor='black', density=True, label='Model')
        ax_hist.hist(real_hardness, bins=50, histtype='step', color='black', linewidth=2, density=True, label='Real')
        ax_hist.set_xlabel('Hardness')
        ax_hist.set_ylabel('Density')

        # ---- Sorted values ----
        sorted_model = np.sort(hardness)
        ax_sorted.plot(sorted_model, label='Model', alpha=0.7)
        ax_sorted.plot(sorted_real, label='Real', color='black', linestyle='--', linewidth=2)
        ax_sorted.set_xlabel('Sample index (sorted)')
        ax_sorted.set_ylabel('Hardness value')

        # ---- KL divergence (Real vs Synthetic) ----
        # Define common bins covering both distributions
        vmin = min(np.min(hardness), np.min(real_hardness))
        vmax = max(np.max(hardness), np.max(real_hardness))
        bins = np.linspace(vmin, vmax, 50)   # 50 bins as in the plots

        # Histograms as counts (not density)
        hist_model, _ = np.histogram(hardness, bins=bins)
        hist_real, _ = np.histogram(real_hardness, bins=bins)

        # Convert counts to probabilities
        prob_model = hist_model / np.sum(hist_model)
        prob_real = hist_real / np.sum(hist_real)

        # Laplace smoothing (add small epsilon) to avoid log(0)
        eps = 1e-10
        prob_model = np.clip(prob_model, eps, 1.0)
        prob_real = np.clip(prob_real, eps, 1.0)

        # KL(real || model)  --  how different is model from real?
        kl_div = np.sum(prob_real * np.log(prob_real / prob_model))

        # Display on the histogram subplot (top‑left corner)
        ax_hist.text(0.05, 0.95, f'KL = {kl_div:.3f}', transform=ax_hist.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        for ax in [ax_hist, ax_sorted]:
            ax.set_title(os.path.basename(fpath).replace('.pkl', ''))

    # Hide any unused subplots
    for ax in axes_hist[n_files:]:
        ax.axis('off')
    for ax in axes_sorted[n_files:]:
        ax.axis('off')

    fig_hist.suptitle(f"Hardness type: {hardness_type} – Histograms", fontsize=14)
    fig_hist.tight_layout(rect=[0, 0, 1, 0.95])
    fig_hist.savefig(f"hardness_histograms_{hardness_type}.png", dpi=150, bbox_inches='tight')

    fig_sorted.suptitle(f"Hardness type: {hardness_type} – Sorted values", fontsize=14)
    fig_sorted.tight_layout(rect=[0, 0, 1, 0.95])
    fig_sorted.savefig(f"hardness_sorted_{hardness_type}.png", dpi=150, bbox_inches='tight')

    plt.show()


def main(dataset_name: str):
    base_dir = os.path.join(ROOT, 'Results', dataset_name, "post_hoc_hardness_estimates")
    hardness_types = ["logit_margins", "softmax_margins", "logit_confidences", "softmax_confidences"]

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

        plot_hardness_for_type(real_hardness, sorted(model_files), hardness_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an ensemble of models on CIFAR-100.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['CIFAR-100'], help='Dataset name: CIFAR-100')

    args = parser.parse_args()
    main(args.dataset_name)
