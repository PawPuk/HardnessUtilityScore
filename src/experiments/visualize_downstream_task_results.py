"""
Produce components of Figure 6 (visualize the results of Case Study 2)
and parts of Figure 8 corresponding to Case Study 2.


Main purpose
-------------------
* Compare Recall and Precision between ensembles trained on: (i) original datasets; and (ii) dataset variants obtained through hardness-based resampling.
* Measure and visualize the changes to fairness coming from hardness-based resampling using gap- and dispersion-based metrics.

Important information
-------------------
* Ensure that `num_models_per_dataset` and `num_datasets` from config.py have the same values as they did during
running experiment3.py!
"""
import argparse
import os

from src.config.config import get_config, ROOT
from src.data.loading import load_real_dataset
from src.models.loading import extract_model_paths
from src.utils.evaluation import obtain_results


class ResamplingVisualizer:
    """Encapsulates all the necessary methods to perform the visualization pertaining to the results of experiment of
    case study 2."""
    def __init__(self, dataset_name):
        """Initialize the Visualizer class responsible for visualizing the improvement in fairness from using different
         hardness-based resampling techniques on the full data.

         :param dataset_name: Name of the dataset
        """
        self.dataset_name = dataset_name

        config = get_config(args.dataset_name)
        self.num_classes = config['num_classes']
        self.num_epochs = config['num_epochs']
        self.num_models_per_dataset = config['num_models_per_dataset']
        self.num_datasets = config['num_datasets']

    def main(self):
        """Main method for producing the visualizations."""
        results_dir = os.path.join(ROOT, "Results", self.dataset_name)
        models = extract_model_paths(self.dataset_name, self.num_epochs, self.num_datasets, self.num_models_per_dataset)
        _, _, test_loader, _ = load_real_dataset(self.dataset_name)

        obtain_results(results_dir, self.num_classes, test_loader, "HBR_results.pkl", models)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load models for specified pruning strategy and dataset")
    parser.add_argument("--dataset_name", type=str, choices=['CIFAR-100'], required=True,
                        help="Name of the dataset (e.g., 'CIFAR10')")

    args = parser.parse_args()
    ResamplingVisualizer(args.dataset_name).main()
