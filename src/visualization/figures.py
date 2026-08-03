import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_fairness(
        fairness_results: Dict[str, Dict[str, Dict[str, Dict[str, Tuple[float, float, List[float]]]]]],
        figure_save_dir: str
):
    for base_metric in ['Precision', 'Recall']:
        baseline_gen, baseline_strat = 'None', 'None'
        baseline_data_by_fairness_metric = fairness_results[base_metric][baseline_gen][baseline_strat]

        generative_models = list(fairness_results[base_metric].keys())
        generative_models = [generative_model for generative_model in generative_models
                             if generative_model != baseline_gen]  # Remove baseline
        strategies = list(fairness_results[base_metric][generative_models[0]].keys())
        fairness_metrics = fairness_results[base_metric][generative_models[0][strategies[0]]]
        for fairness_metric in fairness_metrics:
            baseline_mean = baseline_data_by_fairness_metric[fairness_metric][0]
            labels, means, stds = [], [], []
            for generative_model in generative_models:
                for strategy in strategies:
                    label = f"{generative_model}_{strategy}"
                    labels.append(label)
                    mean, std, _ = fairness_results[base_metric][generative_model][strategy][fairness_metric]
                    means.append(mean)
                    stds.append(std)
            x_positions = np.arange(len(labels))
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.axhline(y=baseline_mean, color='black', linestyle='--', linewidth=2, label=f'Baseline')
            ax.bar(x_positions, means, yerr=stds, capsize=5, color="steelblue", label='Mean ± std')

            ax.set_xticks(x_positions)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel(f"{fairness_metric} value")
            ax.set_title(f"{fairness_metric} based on {base_metric}")
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            plt.tight_layout()
            filename = f"{fairness_metric}_based_on_{base_metric}.pdf"
            plt.savefig(os.path.join(figure_save_dir, filename), dpi=150)
            plt.close()
