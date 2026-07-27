from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config.config import DEVICE
from src.models.neural_networks import ResNet18LowRes


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
    if alpha > 1.0:
        average_sample_count = int(np.mean(samples_per_class))
        for class_id in range(num_classes):
            absolute_difference = abs(samples_per_class[class_id] - average_sample_count)
            if samples_per_class[class_id] > average_sample_count:
                samples_per_class[class_id] = average_sample_count + int(alpha * absolute_difference)
            else:
                samples_per_class[class_id] = average_sample_count - int(alpha * absolute_difference)

    return samples_per_class
