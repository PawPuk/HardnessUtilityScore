import glob
import os
import pickle
from typing import Dict, List, Tuple, Union

from src.config.config import ROOT, get_config


def load_results(path: str):
    """Load results."""
    with open(path, 'rb') as file:
        return pickle.load(file)


def load_previous_hardness_estimates(path: str) -> Union[Dict, Dict[Tuple[int, int], Dict[str, List[float]]]]:
    """Loads the hardness estimates, if they have been computed before, or return an empty Dictionary otherwise."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        prior_hardness_estimates = load_results(path)
        print(f'{path} exists - extended hardness estimates.')
        return prior_hardness_estimates
    else:
        print(f"{path} does not exist or is empty. Initializing new data.")
        return {}


def save_in_hoc_hardness_estimates(in_hoc_hardness_estimates: Dict[Tuple[int, int], Dict[str, List[float]]],
                                   dataset_model_id: Tuple[int, int], dataset_name: str):
    """
    The purpose of this function is to enable easier generation of results. If we already spent a lot of
    resources on training an ensemble, we don't want it to go to waste just because the ensemble is not large
    enough. We want to add more models to the ensemble rather than have to retrain it from scratch.
    """
    hardness_save_dir = os.path.join(ROOT, "Results", dataset_name)
    os.makedirs(hardness_save_dir, exist_ok=True)
    path = os.path.join(hardness_save_dir, 'in_hoc_hardness_estimates.pkl')
    old_in_hoc_hardness_estimates = load_previous_hardness_estimates(path)
    old_in_hoc_hardness_estimates[dataset_model_id] = in_hoc_hardness_estimates[dataset_model_id]

    with open(path, "wb") as file:
        print(f'Saving updated hardness estimates.')
        # noinspection PyTypeChecker
        pickle.dump(old_in_hoc_hardness_estimates, file)


def load_in_hoc_hardness_estimates(dataset_name: str) -> List[List[float]]:
    """Load hardness estimates."""
    path = os.path.join(ROOT, 'Results', dataset_name, 'in_hoc_hardness_estimates.pkl')
    hardness_estimates = load_results(path)
    hardness_over_models = [hardness_estimates[(0, model_id)] for model_id in range(len(hardness_estimates))]
    return hardness_over_models


def extract_paths_to_pretrained_models(dataset_name: str):
    config = get_config(dataset_name)
    base_path = os.path.join(ROOT, config['save_dir'], f"{0.00:.2f}", dataset_name)
    pattern = os.path.join(base_path, f'dataset_*_model_*_epoch_{config["num_epochs"]}.pth')

    model_paths = {}
    for filepath in glob.glob(pattern):
        filename = os.path.basename(filepath)
        # Extract indices: dataset_X_model_Y_epoch_Z.pth
        parts = filename.split('_')
        d_idx = int(parts[1])  # after 'dataset'
        m_idx = int(parts[3])  # after 'model'
        model_paths.setdefault(d_idx, {})[m_idx] = filepath

    return model_paths


def load_sample_allocations(hardness_save_dir: str, dataset_name: str) -> Dict[float, List[int]]:
    """Extracts the sample allocations after resampling for different alphas"""
    sample_allocations = {}
    for root, dirs, files in os.walk(hardness_save_dir):
        if f"{dataset_name}/" in root and 'alpha_' in root:
            alpha = float(root.split('alpha_')[-1])
            for file in files:
                file_path = os.path.join(root, file)
                sample_allocations[alpha] = load_results(file_path)
    print(f'Loaded info on class-wise sample allocation after hardness-based resampling:\n\t{sample_allocations}')
    return sample_allocations
