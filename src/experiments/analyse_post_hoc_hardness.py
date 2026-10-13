import argparse
from glob import glob
import json
import os
import pickle
import shutil
from typing import Dict, List, Tuple
import zipfile

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
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


def plot_hardness_for_type(real_hardness: np.ndarray, model_files: List[str], hardness_type: str,
                           fig_save_dir: str) -> List[np.ndarray]:
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

    kl_divs = []

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
        kl_divs.append(kl_div)

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
    return kl_divs


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
        synthetic_labels: Dict[str, np.ndarray],
        num_classes: int = 100
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    For a given hardness type, compute per-class mean hardness for real and all models,
    then create a figure with one subplot per model.
    In each subplot:
      - A black line shows the real class means (classes sorted by real hardness).
      - Bars show the model's class means.
      - RMSE and RME are displayed in the top‑right corner.
    X‑axis ticks are removed because there are too many classes.
    """
    _, _, _, test_set = load_real_dataset(dataset_name)
    real_labels = load_labels(test_set)
    real_class_means = compute_class_means(real_hardness, real_labels, num_classes)

    model_class_means = []
    for fpath in tqdm(model_files, 'Computing class means over model_files for class-level visualisation'):
        h = compute_avg_hardness_per_model(fpath)
        basename = os.path.basename(fpath)
        generative_model = basename.split('_logit')[0] if 'logit' in basename else basename.split('_softmax')[0]
        means = compute_class_means(h, synthetic_labels[generative_model], num_classes)
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
    rmses, rmes = [], []

    for idx, (ax, fpath) in enumerate(zip(axes, model_files)):
        model_means = model_class_means[idx]
        sorted_model_means = model_means[sorted_classes]

        ax.plot(x_positions, sorted_model_means, color='C0', linestyle='-', linewidth=2, label='Model')
        ax.plot(x_positions, sorted_real_means, color='black', linestyle='-', linewidth=2, label='Real')

        rmse = np.sqrt(np.mean((sorted_real_means - sorted_model_means) ** 2))
        rmses.append(rmse)
        rme = np.sqrt(np.mean(sorted_real_means - sorted_model_means))
        rmes.append(rme)
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
    return rmses, rmes


def plot_metrics(metrics: dict[str, np.ndarray], fig_save_dir: str):
    """Save a six-panel metric figure and Pearson/Spearman correlation matrices."""
    metric_names = list(metrics.keys())
    lengths = {len(v) for v in metrics.values()}
    if len(lengths) != 1:
        raise ValueError(f"Metric lengths do not match: { {name: len(v) for name, v in metrics.items()} }")

    n = next(iter(lengths))
    x = np.arange(n)

    # Five individual plots plus one combined, standardized plot.
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.ravel()

    for i, name in enumerate(metric_names):
        axes[i].plot(x, metrics[name], marker="o", linewidth=1.8)
        axes[i].set_title(name)
        axes[i].set_xlabel("Checkpoints")
        axes[i].set_ylabel(name)
        axes[i].grid(True, linestyle=":", alpha=0.5)

    combined_ax = axes[5]
    for name, y in metrics.items():
        scaled_y = y / np.max(np.abs(y))
        combined_ax.plot(x, scaled_y, marker="o", linewidth=1.8, label=name)

    combined_ax.set_title("All metrics (standardized)")
    combined_ax.set_xlabel("Checkpoint")
    combined_ax.set_ylabel("Z-score")
    combined_ax.grid(True, linestyle=":", alpha=0.5)
    combined_ax.legend(fontsize=9)

    fig.suptitle("Evaluation metrics", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(fig_save_dir, "metrics_over_checkpoints.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_correlations(metrics: dict[str, np.ndarray], fig_save_dir: str):
    # Each row represents a checkpoint; each column a metric.
    metric_names = list(metrics.keys())
    data = np.column_stack([metrics[name] for name in metric_names])

    for method, corr_func in [("pearson", pearsonr), ("spearman", spearmanr)]:
        corr = np.empty((len(metric_names), len(metric_names)))
        p_vals = np.empty_like(corr)

        for i in range(len(metric_names)):
            for j in range(len(metric_names)):
                if i == j:
                    corr[i, j] = 1.0
                    p_vals[i, j] = 0.0
                elif method == "pearson":
                    result = corr_func(data[:, i], data[:, j])
                    corr[i, j], p_vals[i, j] = result.statistic, result.pvalue
                else:
                    result = corr_func(data[:, i], data[:, j])
                    corr[i, j], p_vals[i, j] = result.statistic, result.pvalue

        fig, ax = plt.subplots(figsize=(9, 7))
        image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        fig.colorbar(image, ax=ax, label="Correlation coefficient")

        ax.set_xticks(range(len(metric_names)), metric_names, rotation=35, ha="right")
        ax.set_yticks(range(len(metric_names)), metric_names)

        for i in range(len(metric_names)):
            for j in range(len(metric_names)):
                ax.text(j, i, f"r={corr[i, j]:.2f}\np={p_vals[i, j]:.3g}", ha="center", va="center", fontsize=8,
                        color="white" if abs(corr[i, j]) > 0.55 else "black")

        ax.set_title(f"{method.capitalize()} correlations")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_save_dir, f"{method}_correlation_matrix.png"), dpi=180,  bbox_inches="tight")
        plt.close(fig)



def main(dataset_name: str):
    base_dir = os.path.join(ROOT, 'Results', dataset_name, "post_hoc_hardness_estimates")
    hardness_types = ["softmax_margins", "softmax_confidences", "logit_margins", "logit_confidences"]

    all_files = glob(os.path.join(base_dir, "*.pkl"))
    if not all_files:
        raise Exception(f"No .pkl files found in {base_dir}")

    syn_label_save_path = os.path.join(ROOT, 'synthetic_data', dataset_name, 'synthetic_labels.json')
    if os.path.exists(syn_label_save_path):
        synthetic_labels = np.load(syn_label_save_path)
    else:
        synthetic_labels = {}
    for f in tqdm(all_files, desc='Extracting synthetic labels'):
        basename = os.path.basename(f)
        generative_model = basename.split('_logit')[0] if 'logit' in basename else basename.split('_softmax')[0]

        if 'metric' or 'real' in f or generative_model in synthetic_labels.keys():
            continue

        synth_root = os.path.join(ROOT, 'synthetic_data', dataset_name)
        with zipfile.ZipFile(os.path.join(synth_root, f"{generative_model}.zip"), 'r') as zip_ref:
            zip_ref.extractall(synth_root)

        # In the below the value of normalize doesn't matter as synthetic_set is only used to extract labels
        _, synthetic_set = load_synthetic_dataset(dataset_name, generative_model, False)
        synthetic_labels[generative_model] = load_labels(synthetic_set)
        with open(syn_label_save_path, 'w') as f:
            json.dump(synthetic_labels, f)

        shutil.rmtree(os.path.join(synth_root, generative_model))

    for hardness_type in tqdm(hardness_types):
        for generative_architecture in ['stylegan', 'edm']:
            with zipfile.ZipFile(os.path.join(ROOT, 'synthetic_data', dataset_name, f'{generative_model}_metrics.zip'),
                                 'r') as zip_ref:
                zip_ref.extractall(os.path.join(ROOT, 'synthetic_data', dataset_name))

            # Separate the real_test file for this type and model files
            real_file, model_files, snapshot_files, IS_metrics, FID_metrics = None, [], [], [], []
            for f in all_files:
                basename = os.path.basename(f)
                if generative_architecture in basename:
                    snapshot_idx = basename.split("_")[1]
                    snapshot_files.append(f'snapshot_{snapshot_idx}.json')
                if hardness_type in basename:
                    if basename == f"real_test_{hardness_type}.pkl":
                        real_file = f
                    elif generative_architecture in basename:
                        model_files.append(f)
            real_hardness = compute_avg_hardness_per_model(real_file)
            model_files = sort_models(model_files)

            for snapshot_file in snapshot_files:
                with open(os.path.join(ROOT, 'synthetic_data', dataset_name, f'{generative_architecture}_metrics',
                                       snapshot_file)) as f:
                    d = json.load(f)
                    IS_metrics.append(d['Inception_Score_Mean'])
                    FID_metrics.append(d['Frechet_Inception_Distance'])

            fig_save_dir = os.path.join(ROOT, 'Figures', dataset_name, generative_architecture, hardness_type)
            os.makedirs(fig_save_dir, exist_ok=True)
            kl_divs = plot_hardness_for_type(real_hardness, model_files, hardness_type, fig_save_dir)
            rmses, rmes = plot_class_hardness(dataset_name, real_hardness, model_files, hardness_type, fig_save_dir,
                                              synthetic_labels)

            metrics = {"RMSE": np.asarray(rmses), "RME": np.asarray(rmes), "KL_DIV": np.asarray(kl_divs),
                       "IS": np.asarray(IS_metrics), "FID": np.asarray(FID_metrics)}
            plot_metrics(metrics, fig_save_dir)
            plot_correlations(metrics, fig_save_dir)

        shutil.rmtree(os.path.join(ROOT, 'synthetic_data', dataset_name, f'{generative_architecture}_metrics'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare hardness distributions between real and synthetic data.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['CIFAR-100'], help='Dataset name: CIFAR-100')

    args = parser.parse_args()
    main(args.dataset_name)
