from typing import Any, Dict, List, Tuple, Union

import torch
import torchvision

from src.config.config import DEVICE
from src.models.neural_networks import ResNet18LowRes


def compute_AUM(
        batch_indices: torch.Tensor,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        in_hoc_hardness_estimates: Dict[Tuple[int, int], List[List[float]]],
        epoch: int,
        dataset_model_id: Tuple[int, int]
):
    """Estimate in-hoc hardness through AUM (https://arxiv.org/pdf/2001.10528)."""

    for index_within_batch, (i, logits, correct_label) in enumerate(zip(batch_indices, outputs, labels)):
        i = i.item()
        correct_label = correct_label.item()
        logits = logits.detach()
        correct_logit = logits[correct_label].item()

        max_other_logit = torch.max(torch.cat((logits[:correct_label], logits[correct_label + 1:]))).item()
        in_hoc_hardness_estimates[dataset_model_id][i][epoch] = correct_logit - max_other_logit


def compute_confidences(
        model_states: List[Any],
        images: List[torch.Tensor],
        class_id: int,
        num_classes: int,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        batch_size: int = 1024
) -> List[float]:
    """Estimate hardness through confidence. This is used in data-resampling.py when using hEDM or aEDM to estimate the
    hardness of real and synthetic samples"""
    num_samples, avg_confidences = len(images), []

    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_images = images[batch_start:batch_end]

        normalize = torchvision.transforms.Normalize(mean=mean, std=std)
        normalized_images = [normalize(img) for img in batch_images]
        batch_normalized_images = torch.stack(normalized_images).to(DEVICE)  # Shape: [B, 3, 32, 32]

        # For each model, compute confidence
        batch_confidences = torch.zeros(batch_normalized_images.size(0), device=DEVICE)
        for model_state in model_states:
            model = ResNet18LowRes(num_classes)
            model.load_state_dict(model_state)
            model = model.to(DEVICE)
            model.eval()
            with torch.no_grad():
                logits = model(batch_normalized_images)
                probs = torch.nn.functional.softmax(logits, dim=1)
                conf = probs[:, class_id]  # confidence for true class
                batch_confidences += conf  # accumulate per model

        batch_confidences /= len(model_states)  # average confidence across models
        avg_confidences.extend(batch_confidences.cpu().tolist())

    return avg_confidences
