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
import scipy
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


def load_IS_and_FID(
        dataset_name: str,
        gen_architecture: str,
        snapshot_files: List[str]
) -> Tuple[List[float], List[float]]:
    IS_metrics, FID_metrics = [], []
    zip_file_path = os.path.join(ROOT, 'synthetic_data', dataset_name, f'{gen_architecture}_metrics.zip')
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(ROOT, 'synthetic_data', dataset_name))
    for snapshot_file in snapshot_files:
        with open(os.path.join(ROOT, 'synthetic_data', dataset_name, f'{gen_architecture}_metrics',
                               snapshot_file)) as f:
            d = json.load(f)
            IS_metrics.append(d['Inception_Score_Mean'])
            FID_metrics.append(d['Frechet_Inception_Distance'])
    shutil.rmtree(os.path.join(ROOT, 'synthetic_data', dataset_name, f'{gen_architecture}_metrics'))
    return IS_metrics, FID_metrics


def sort_models(file_list: List[str]) -> List[str]:
    """Sort files by the sorting_type."""
    return sorted(file_list)


def plot_hardness_for_type(
        real_hardness: np.ndarray,
        model_files: List[str],
        hardness_type: str,
        fig_save_dir: str
) -> List[np.ndarray]:
    """
    Computes KL divergence for all checkpoints and generates a stacked Ridge Plot.
    """
    kl_divs, checkpoint_hardness_data, curves, eps = [], [], [], 1e-10

    # Common bounds for density histograms
    vmin = np.min(real_hardness)
    vmax = np.max(real_hardness)

    for fpath in model_files:
        hardness = compute_avg_hardness_per_model(fpath)
        checkpoint_hardness_data.append((fpath, hardness))
        vmin = min(vmin, np.min(hardness))
        vmax = max(vmax, np.max(hardness))

    bins = np.linspace(vmin, vmax, 100)
    hist_real, _ = np.histogram(real_hardness, bins=bins, density=True)
    prob_real = hist_real / np.sum(hist_real)

    # Compute KL divergence for all checkpoints
    for _, hardness in checkpoint_hardness_data:
        hist_model, _ = np.histogram(hardness, bins=bins, density=True)
        prob_model = hist_model / np.sum(hist_model)
        prob_model = (prob_model + eps) / np.sum(prob_model + eps)
        kl_div = scipy.stats.entropy(prob_real, prob_model)
        kl_divs.append(kl_div)

    # ---- Ridge Plot for milestone checkpoints ----
    indices_to_plot = [len(model_files) - 1, 14, 4, 0]
    fig, ax = plt.subplots(figsize=(10, 6))

    x_vals = 0.5 * (bins[:-1] + bins[1:])
    real_density, _ = np.histogram(real_hardness, bins=bins, density=True)

    max_density = max(np.max(real_density),
                      max([np.max(np.histogram(h, bins=bins, density=True)[0]) for _, h in checkpoint_hardness_data]))
    spacing = max_density * 0.65

    # Bottom reference: Real Data
    curves.append(('Real Data (Target)', real_density, 'black', 0.0))
    for rank, idx in enumerate(indices_to_plot):
        fpath, hardness = checkpoint_hardness_data[idx]
        epoch_label = os.path.basename(fpath).replace('.pkl', '').split('_')[1]
        kl_val = kl_divs[idx]
        density, _ = np.histogram(hardness, bins=bins, density=True)
        y_offset = (rank + 1) * spacing
        curves.append((f'Checkpoint {idx + 1} (Epoch {epoch_label}, KL={kl_val:.3f})', density, f'C{rank}', y_offset))

    for label, density, color, y_offset in curves:
        y_vals = density + y_offset
        ax.plot(x_vals, y_vals, color=color, linewidth=1.5 if color != 'black' else 2.5, label=label)
        ax.fill_between(x_vals, y_offset, y_vals, color=color, alpha=0.3 if color != 'black' else 0.1)

    ax.set_xlabel('Hardness', fontsize=12)
    ax.set_ylabel('Density (Stacked over Training)', fontsize=12)
    ax.set_title(f'Hardness Spectrum Evolution (Ridge Plot) – {hardness_type}', fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.4, axis='x')

    fig.tight_layout()
    fig.savefig(os.path.join(fig_save_dir, "hardness_ridge_plot.png"), dpi=180, bbox_inches='tight')
    plt.close(fig)

    return kl_divs


def load_labels(dataset: IndexedDataset) -> np.ndarray:
    labels = []
    for i in range(len(dataset)):
        labels.append(dataset[i][1])
    return np.array(labels)


def compute_class_means(hardness: np.ndarray, labels: np.ndarray, num_classes: int = 100) -> np.ndarray:
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
) -> List[np.ndarray]:
    """
    Computes class-wise means across all models (returning mean_errors for all checkpoints),
    but generates a single clean plot displaying the Real Data line and 4 milestone checkpoints.
    """
    _, _, _, test_set = load_real_dataset(dataset_name)
    real_labels = load_labels(test_set)
    real_class_means = compute_class_means(real_hardness, real_labels, num_classes)

    model_class_means = []
    for fpath in tqdm(model_files):
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
    sorted_classes = valid_classes[sorted_indices]
    sorted_real_means = real_class_means[sorted_classes]

    x_positions = np.arange(len(sorted_classes))
    mean_errors = []

    # Calculate mean errors for all checkpoints (required for correlation matrices & metric tracking)
    for idx, fpath in enumerate(model_files):
        model_means = model_class_means[idx]
        sorted_model_means = model_means[sorted_classes]
        errors = sorted_model_means - sorted_real_means
        mean_error = np.mean(errors)
        mean_errors.append(mean_error)

    # ---- Single Consolidated Milestone Plot for Class Hardness Spectra ----
    indices_to_plot = [0, 4, len(model_files) - 1]
    linestyles = [':', '--', '-.']

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_positions, sorted_real_means, color='black', linestyle='-', linewidth=2.5, label='Real Data (Target)')
    for rank, idx in enumerate(indices_to_plot):
        model_means = model_class_means[idx]
        sorted_model_means = model_means[sorted_classes]
        ax.plot(x_positions, sorted_model_means, linestyle=linestyles[rank], linewidth=2.5,
                label=f'Checkpoint {idx + 1}')

    ax.set_xticks(np.arange(0, len(sorted_classes), 10))
    ax.set_xlabel('Class ID (sorted by real hardness)', fontsize=12)
    ax.set_ylabel('Mean Class Hardness', fontsize=12)
    ax.set_title(f'Class-wise Hardness Spectra Progression – {hardness_type}', fontsize=14)
    ax.legend(loc='upper left', fontsize=14, framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.4)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_save_dir, f"class_hardness_spectra.png"), dpi=180, bbox_inches='tight')
    plt.close(fig)

    return mean_errors


def plot_correlations(metrics: dict[str, np.ndarray], fig_save_dir: str):
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
                else:
                    result = corr_func(data[:, i], data[:, j])
                    corr[i, j], p_vals[i, j] = result.statistic, result.pvalue

        fig, ax = plt.subplots(figsize=(9, 7))
        image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        fig.colorbar(image, ax=ax, label="Correlation coefficient")

        ax.set_xticks(range(len(metric_names)), metric_names)
        ax.set_yticks(range(len(metric_names)), metric_names)

        for i in range(len(metric_names)):
            for j in range(len(metric_names)):
                ax.text(j, i, f"r={corr[i, j]:.2f}\np={p_vals[i, j]:.3g}", ha="center", va="center", fontsize=8,
                        color="white" if abs(corr[i, j]) > 0.55 else "black")

        ax.set_title(f"{method.capitalize()} correlations")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_save_dir, f"{method}_correlation_matrix.png"), dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_metrics(metrics: Dict[str, Dict[str, np.ndarray]], fig_save_dir: str):
    global_lenghts = []
    for hardness_type in metrics.keys():
        lengths = {len(v) for v in metrics[hardness_type].values()}
        global_lenghts.append(lengths)
        if len(lengths) != 1:
            raise ValueError(f"Metric lengths do not match with {hardness_type}: "
                             f"{ {name: len(v) for name, v in metrics[hardness_type].items()} }")
    if any(s != global_lenghts[0] for s in global_lenghts):
        raise ValueError(f'Metric lengths do not match across hardness types: {global_lenghts}.')

    n = global_lenghts[0].pop()
    x = np.arange(n)

    fig, axes = plt.subplots(2, 2, figsize=(18, 9))
    axes = axes.ravel()

    for i, hardness_type in enumerate(metrics.keys()):
        for metric_name, y in metrics[hardness_type].items():
            scaled_y = y / np.max(np.abs(y))
            axes[i].plot(x, scaled_y, marker='o', linewidth=2, label=metric_name)
            axes[i].set_title(f'Standardized metrics for {hardness_type}')
            axes[i].set_xlabel('Checkpoints')
            axes[i].set_ylabel('Z-score')
            axes[i].grid(True, linestyle=':', alpha=0.5)
            axes[i].legend(fontsize=9)

    fig.suptitle("Evaluation metrics", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(fig_save_dir, "metrics_over_checkpoints.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(dataset_name: str):
    base_dir = os.path.join(ROOT, 'Results', dataset_name, "post_hoc_hardness_estimates")
    hardness_types = ["softmax_margins", 'logit_margins', 'softmax_confidences', 'logit_confidences']

    all_files = sorted(glob(os.path.join(base_dir, "*.pkl")))
    if not all_files:
        raise Exception(f"No .pkl files found in {base_dir}")

    for gen_architecture in ['edm', 'stylegan']:
        metrics = {}
        for hardness_type in tqdm(hardness_types):
            real_file, model_files, snapshot_files, metrics[hardness_type] = None, [], [], {}
            for f in all_files:
                basename = os.path.basename(f)
                if gen_architecture in basename:
                    snapshot_idx = basename.split("_")[1]
                    if f'snapshot_{snapshot_idx}.json' not in snapshot_files:
                        snapshot_files.append(f'snapshot_{snapshot_idx}.json')
                if hardness_type in basename:
                    if basename == f"real_test_{hardness_type}.pkl":
                        real_file = f
                    elif gen_architecture in basename:
                        model_files.append(f)
            real_hardness = compute_avg_hardness_per_model(real_file)

            IS_metrics, FID_metrics = load_IS_and_FID(dataset_name, gen_architecture, snapshot_files)

            model_files = sort_models(model_files)
            fig_save_dir = os.path.join(ROOT, 'Figures', dataset_name, gen_architecture, hardness_type)
            os.makedirs(fig_save_dir, exist_ok=True)

            kl_divs = plot_hardness_for_type(real_hardness, model_files, hardness_type, fig_save_dir)
            mean_errors = plot_class_hardness(dataset_name, real_hardness, model_files, hardness_type, fig_save_dir)
            print(gen_architecture, hardness_type, kl_divs, mean_errors)

            metrics[hardness_type] = {'ME': np.asarray(mean_errors), "KL_DIV": np.asarray(kl_divs),
                                      "IS": np.asarray(IS_metrics), "FID": np.asarray(FID_metrics)}
            plot_correlations(metrics[hardness_type], fig_save_dir)
        fig_save_dir = os.path.join(ROOT, 'Figures', dataset_name, gen_architecture)
        plot_metrics(metrics, fig_save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare hardness distributions between real and synthetic data.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['CIFAR-100'], help='Dataset name: CIFAR-100')

    args = parser.parse_args()
    main(args.dataset_name)
