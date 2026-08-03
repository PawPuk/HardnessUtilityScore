from collections import defaultdict
import os
from typing import Dict, List, Tuple

import dill
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.config import DEVICE
from src.models.neural_networks import ResNet18LowRes
from src.utils.structures import defaultdict_to_dict


def evaluate_model(
        model: ResNet18LowRes,
        criterion: nn.CrossEntropyLoss,
        test_loader: DataLoader
) -> Tuple[float, float]:
    """Evaluate the model on the test set."""
    model.eval()
    correct, total, running_loss = 0, 0, 0.0

    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = running_loss / total
    return avg_loss, accuracy


def compute_sample_allocation_for_resampling(
        hardness_scores: List[float],
        labels: List[int],
        num_classes: int,
        num_training_samples: int,
        alpha: float = 1.0
) -> List[int]:
    """Compute number of samples per class after hardness-based resampling according to hardness_scores."""
    # Divide the instant-level hardness estimates into classes.
    hardness_by_class = {class_id: [] for class_id in range(num_classes)}
    for i, label in enumerate(labels):
        hardness_by_class[label].append((i, hardness_scores[i]))

    # Compute (or extract) average hardness of each class
    class_hardness = {class_id: np.mean([score for _, score in entries])
                      for class_id, entries in hardness_by_class.items()}

    # Add offset in case some classes have negative hardness values to not get nonsensical resampling ratios.
    if min(class_hardness.values()) < 0:
        offset = -min(class_hardness.values())
        for class_id in range(num_classes):
            class_hardness[class_id] += offset + 0.0001  # Adding epsilon to not divide by zero later on.

    # Compute the resampling ratios for each class.
    hardness_ratios = {class_id: 1 / float(val) for class_id, val in class_hardness.items()}
    ratios = {class_id: class_hardness / sum(hardness_ratios.values())
              for class_id, class_hardness in hardness_ratios.items()}

    # Compute the amount of samples per class after resampling.
    samples_per_class = [int(round(ratio * num_training_samples)) for ratio in ratios.values()]

    # Tailor the degree of the introduces data imbalance (only applicable if alpha is larger than 1).
    average_sample_count = int(np.mean(samples_per_class))
    for class_id in range(num_classes):
        absolute_difference = abs(samples_per_class[class_id] - average_sample_count)
        if samples_per_class[class_id] > average_sample_count:
            samples_per_class[class_id] = average_sample_count + int(alpha * absolute_difference)
        else:
            samples_per_class[class_id] = average_sample_count - int(alpha * absolute_difference)

    return samples_per_class


def evaluate_ensembles(
        ensembles: Dict[int, List[str]],
        test_loader: DataLoader,
        class_index: int,
        num_classes: int,
        generative_model: str,
        strategy: str,
        results: Dict[str, Dict[str, Dict[str, Dict[int, Dict[int, Dict[int, float]]]]]]
):
    """Used to compute the scores necessary to produce visualizations - Tp, Tn, Fp, Fn, Precision, and Recall."""
    for dataset_idx in range(len(ensembles)):
        for model_idx, model_path in enumerate(ensembles[dataset_idx]):
            model_state = torch.load(model_path)
            true_positives, true_negatives, false_positives, false_negatives = 0, 0, 0, 0
            model = ResNet18LowRes(num_classes)
            model.load_state_dict(model_state)
            model = model.to(DEVICE)
            model.eval()

            with torch.no_grad():
                for images, labels, _ in test_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    _, predicted = outputs.max(1)

                    for pred, label in zip(predicted, labels):
                        if label == class_index:
                            if pred.item() == class_index:
                                true_positives += 1
                            else:
                                false_negatives += 1
                        else:
                            if pred.item() == class_index:
                                false_positives += 1
                            else:
                                true_negatives += 1

            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)

            for (metric_name, metric_results) in [('Tp', true_positives), ('Tn', true_negatives), ('Recall', recall),
                                                  ('Fp', false_positives), ('Fn', false_negatives),
                                                  ('Precision', precision)]:
                results[metric_name][generative_model][strategy][class_index][dataset_idx][model_idx] = metric_results


def obtain_results(
        save_dir: str,
        num_classes: int,
        test_loader: DataLoader,
        file_name: str,
        models: Dict[str, Dict[str, Dict[int, List[str]]]],
) -> Dict[str, Dict[str, Dict[str, Dict[int, Dict[int, Dict[int, float]]]]]]:
    """Load the results if they have been computed before, or use evaluate_ensembles to compute them."""
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: defaultdict(float))))))
    if os.path.exists(os.path.join(save_dir, file_name)):
        print('Loading pre-computed ensemble results.')
        with open(os.path.join(save_dir, file_name), 'rb') as f:
            results = dill.load(f)
    else:
        for class_index in tqdm(range(num_classes), desc='Iterating through classes'):
            for generative_model, ensembles_over_strategies in models.items():
                for strategy, ensembles in ensembles_over_strategies.items():
                    evaluate_ensembles(ensembles, test_loader, class_index, num_classes, generative_model, strategy,
                                       results)
        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, file_name), "wb") as file:
            dill.dump(results, file)
    print('The loaded results have keys:', results.keys())
    return defaultdict_to_dict(results)
