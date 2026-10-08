import argparse
from glob import glob
import os
import pickle
import shutil
from typing import List
import zipfile

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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


def sort_models(file_list: List[str]) -> List[str]:
    """Sort files by the sorting_type."""
    return sorted(file_list)


def plot_hardness_for_type(
        real_hardness: np.ndarray,
        model_files: List[str],
        hardness_type: str,
        fig_save_dir: str
) -> None:
    """
    For a given hardness type (e.g., 'logit_margins'), find all model .pkl files
    that contain that type in their name (excluding real_test files), average over
    model indices, and plot a histogram for each model. If a real_test file
    named `real_test_{hardness_type}.pkl` exists, overlay its hardness distribution.
    """
    n_files = len(model_files)
    cols = min(4, n_files)
    rows = (n_files + cols - 1) // cols

    fig_hist, axes_hist = plt.subplots(rows, cols, figsize=(16, 10))
    axes_hist = axes_hist.flatten()

    # For each model file, compute averaged hardness and plot histogram
    for ax_hist, fpath in zip(axes_hist, model_files):
        hardness = compute_avg_hardness_per_model(fpath)

        # ---- Histogram (density) ----
        ax_hist.hist(hardness, bins=50, alpha=0.7, edgecolor='black', density=True, label='Model')
        ax_hist.hist(real_hardness, bins=50, histtype='step', color='black', linewidth=2, density=True, label='Real')
        ax_hist.set_xlabel('Hardness')
        ax_hist.set_ylabel('Density')

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
        ax_hist.set_title(os.path.basename(fpath).replace('.pkl', ''))

    # Hide any unused subplots
    for ax in axes_hist[n_files:]:
        ax.axis('off')

    fig_hist.suptitle(f"Hardness type: {hardness_type} – Histograms", fontsize=14)
    fig_hist.tight_layout(rect=[0, 0, 1, 0.95])
    fig_hist.savefig(os.path.join(fig_save_dir, f"hardness_histograms.png"), dpi=150, bbox_inches='tight')


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
        fig_save_dir: str,
        num_classes: int = 100
) -> None:
    """
    For a given hardness type, compute per-class mean hardness for real and all models,
    then create a figure with one subplot per model.
    In each subplot:
      - A black line shows the real class means (classes sorted by real hardness).
      - Bars show the model's class means.
      - NRMSE is displayed in the top‑right corner.
    X‑axis ticks are removed because there are too many classes.
    """
    _, _, _, test_set = load_real_dataset(dataset_name)
    real_labels = load_labels(test_set)
    real_class_means = compute_class_means(real_hardness, real_labels, num_classes)

    model_class_means = []
    for fpath in model_files:
        h = compute_avg_hardness_per_model(fpath)
        basename = os.path.basename(fpath)
        generative_model = basename.split('_logit')[0] if 'logit' in basename else basename.split('_softmax')[0]

        synth_root = os.path.join(ROOT, 'synthetic_data', dataset_name)
        with zipfile.ZipFile(os.path.join(synth_root, f"{generative_model}.zip"), 'r') as zip_ref:
            zip_ref.extractall(synth_root)

        _, synthetic_set = load_synthetic_dataset(dataset_name, generative_model, False)
        synthetic_labels = load_labels(synthetic_set)
        means = compute_class_means(h, synthetic_labels, num_classes)
        model_class_means.append(means)

        shutil.rmtree(os.path.join(synth_root, generative_model))

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

        ax.plot(x_positions, sorted_model_means, color='C0', linestyle='-', linewidth=2, label='Model')
        ax.plot(x_positions, sorted_real_means, color='black', linestyle='-', linewidth=2, label='Real')

        rmse = np.mean((sorted_real_means - sorted_model_means) ** 2)
        rme = np.mean(sorted_real_means - sorted_model_means)
        ax.text(0.95, 0.95, f'rmse = {rmse:.3f}\nrme = {rme:.3f}', transform=ax.transAxes, verticalalignment='top',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

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
    fig.savefig(os.path.join(fig_save_dir, f"class_hardness_spectra.png"), dpi=150, bbox_inches='tight')


def main(dataset_name: str):
    base_dir = os.path.join(ROOT, 'Results', dataset_name, "post_hoc_hardness_estimates")
    hardness_types = ["softmax_margins", "softmax_confidences", "logit_margins", "logit_confidences"]

    all_files = glob(os.path.join(base_dir, "*.pkl"))
    if not all_files:
        raise Exception(f"No .pkl files found in {base_dir}")

    for generative_model in ['stylegan', 'edm']:
        for hardness_type in tqdm(hardness_types):
            # Separate the real_test file for this type and model files
            real_file, model_files = None, []
            for f in all_files:
                basename = os.path.basename(f)
                if hardness_type in basename:
                    if basename == f"real_test_{hardness_type}.pkl":
                        real_file = f
                    elif generative_model in basename:
                        model_files.append(f)
            real_hardness = compute_avg_hardness_per_model(real_file)

            model_files = sort_models(model_files)
            fig_save_dir = os.path.join(ROOT, 'Figures', dataset_name, generative_model, hardness_type)
            os.makedirs(fig_save_dir, exist_ok=True)
            plot_hardness_for_type(real_hardness, model_files, hardness_type, fig_save_dir)
            plot_class_hardness(dataset_name, real_hardness, model_files, hardness_type, fig_save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare hardness distributions between real and synthetic data.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['CIFAR-100'], help='Dataset name: CIFAR-100')

    args = parser.parse_args()
    main(args.dataset_name)
