import argparse
from glob import glob
import os
import pickle
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from src.config.config import ROOT
from src.data.datasets import IndexedDataset
from src.data.loading import load_real_dataset, load_synthetic_dataset


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


def load_labels(dataset: IndexedDataset) -> np.ndarray:
    """
    Load the test set labels for the given dataset.Returns a 1D numpy array of integer labels (0-99) in the same order as the hardness arrays.
    """
    labels = []
    for i in range(len(dataset)):
        labels.append(dataset[i][1])
    return np.array(labels)


def compute_class_means(hardness: np.ndarray, labels: np.ndarray, num_classes: int = 100) -> np.ndarray:
    """
    Given hardness per sample and corresponding class labels, compute mean hardness per class.
    Classes with no samples get NaN.
    """
    class_means = np.full(num_classes, np.nan)

    for c in range(num_classes):
        mask = (labels == c)
        if np.any(mask):
            class_means[c] = np.mean(hardness[mask])
    return class_means


def plot_class_hardness(
        dataset_name: str,
        real_hardness: np.ndarray,
        model_files: List[str],
        hardness_type: str,
        num_classes: int = 100
) -> None:
    """
    For a given hardness type, compute per-class mean hardness for real and all models,
    then create two figures:
      1) Bar charts: one subplot per model, showing real vs model class means.
      2) Line plot: all models + real over classes sorted by real hardness.
    Classes are sorted by real hardness (ascending). Missing classes (NaNs) are left as gaps.
    """
    _, _, _, test_set = load_real_dataset(dataset_name)
    real_labels = load_labels(test_set)
    real_class_means = compute_class_means(real_hardness, real_labels, num_classes)

    model_class_means = []
    for fpath in model_files:
        h = compute_avg_hardness_per_model(fpath)
        basename = os.path.basename(fpath)
        generative_model = basename.split('_logit')[0] if 'logit' in basename else basename.split('_softmax')[0]
        _, synthetic_set = load_synthetic_dataset(dataset_name, generative_model, False)
        synthetic_labels = load_labels(synthetic_set)
        means = compute_class_means(h, synthetic_labels, num_classes)
        model_class_means.append(means)

    valid_real = ~np.isnan(real_class_means)
    sorted_indices = np.argsort(real_class_means[valid_real])
    valid_classes = np.arange(num_classes)[valid_real]
    sorted_classes = valid_classes[sorted_indices]  # class IDs in ascending real hardness
    sorted_real_means = real_class_means[sorted_classes]

    n_files = len(model_files)
    cols = min(4, n_files)
    rows = (n_files + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, 12))
    axes = axes.flatten()

    x_positions = np.arange(len(sorted_classes))   # x coordinates for bars

    for idx, (ax, fpath) in enumerate(zip(axes, model_files)):
        model_means = model_class_means[idx]
        sorted_model_means = model_means[sorted_classes]

        ax.bar(x_positions, sorted_model_means, width=0.6, color='C0', alpha=0.7, label='Model')
        ax.plot(x_positions, sorted_real_means, color='black', linestyle='-', linewidth=2, label='Real')

        ax.set_xticks([])
        ax.set_xlabel('Class ID (sorted by real hardness)')
        ax.set_ylabel('Mean Hardness')
        ax.set_title(os.path.basename(fpath).replace('.pkl', ''))
        ax.legend()
        ax.grid(True, axis='y', linestyle=':', alpha=0.4)

    for ax in axes[n_files:]:
        ax.axis('off')

    fig.suptitle(f"Class-wise Hardness – {hardness_type} (real line + model bars)", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"class_hardness_bars_{hardness_type}.png", dpi=150, bbox_inches='tight')
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
        sorted_models = sorted(model_files)

        plot_hardness_for_type(real_hardness, sorted_models, hardness_type)
        plot_class_hardness(dataset_name, real_hardness, sorted_models, hardness_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare hardness distributions between real and synthetic data.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['CIFAR-100'], help='Dataset name: CIFAR-100')

    args = parser.parse_args()
    main(args.dataset_name)
