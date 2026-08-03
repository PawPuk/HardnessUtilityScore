from collections import defaultdict
import os
from typing import Dict, List

from src.config.config import ROOT
from src.utils.structures import defaultdict_to_dict


def extract_baseline_model_paths(
        dataset_name: str,
        num_epochs: int,
        models: Dict[str, Dict[str, Dict[int, List[str]]]]
):
    """Loads the baseline models."""
    models_dir = os.path.join(ROOT, "Models/")
    full_dataset_dir = os.path.join(models_dir, "0.00", dataset_name)
    if os.path.exists(full_dataset_dir):
        for file in os.listdir(full_dataset_dir):
            if file.endswith(".pth") and f"_epoch_{num_epochs}" in file:
                model_path = os.path.join(full_dataset_dir, file)
                models['None']['None'][0].append(model_path)


def extract_model_paths(
        dataset_name: str,
        num_epochs: int,
        num_datasets: int,
        num_models_per_dataset: int
) -> Dict[str, Dict[str, Dict[int, List[str]]]]:
    """Used to load the trained models."""
    models_dir = os.path.join(ROOT, "Models")
    models = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for root, dirs, files in os.walk(models_dir):
        if os.path.basename(root) == dataset_name:
            parent_dir = os.path.basename(os.path.dirname(root))
            parts = parent_dir.split('_')
            if len(parts) == 3:
                strategy = parts[0]
                generative_model = parts[1]
            else:
                continue

            for file in files:
                if file.endswith(".pth") and f"_epoch_{num_epochs}" in file:
                    dataset_index = int(file.split("_")[1])
                    model_index = int(file.split("_")[3])
                    if dataset_index >= num_datasets or model_index >= num_models_per_dataset:
                        raise Exception('The `num_datasets` and `num_models_per_dataset` in config.py needs to '
                                        'have the same values as it when running experiment3.py.')

                    model_path = os.path.join(root, file)
                    models[generative_model][strategy][dataset_index].append(model_path)
            if len(models[generative_model][strategy]) > 0:
                print(f"Loaded {len(models[generative_model][strategy])} ensembles of models for "
                      f"{generative_model} and {strategy} oversampling, with each ensemble having "
                      f"{len(models[generative_model][strategy][0])} models.")
                for i in range(1, len(models[generative_model][strategy])):  # Sanity check
                    assert len(models[generative_model][strategy][0]) == len(models[generative_model][strategy][i])

    extract_baseline_model_paths(dataset_name, num_epochs, models)
    return defaultdict_to_dict(models)
