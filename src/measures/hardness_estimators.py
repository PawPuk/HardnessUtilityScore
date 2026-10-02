from typing import Dict, List, Tuple

import torch

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


def compute_margins_and_confidences(
    model: ResNet18LowRes,
    data_loader: torch.utils.data.DataLoader,
) -> List[float]:
    """
    Compute the margin for each sample.

    Returns a list of margins (one per sample) in the same order as the data_loader.
    """
    logit_margins, softmax_margins, all_logit_confidences, all_softmax_confidences = [], [], [], []

    with torch.no_grad():
        for images, labels, _ in data_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)  # [batch_size, num_classes]
            logit_scores = logits
            softmax_scores = torch.softmax(logits, dim=1)

            # Scores for the true class
            correct_logit_scores = logit_scores.gather(1, labels.unsqueeze(1)).squeeze(1)
            correct_softmax_scores = logit_scores.gather(1, labels.unsqueeze(1)).squeeze(1)

            # Mask out the true class to get max among all others
            masked_logit_scores = logit_scores.clone()
            masked_softmax_scores = softmax_scores.clone()
            masked_logit_scores.scatter_(1, labels.unsqueeze(1), -float('inf'))
            masked_softmax_scores.scatter_(1, labels.unsqueeze(1), -float('inf'))
            max_other_logit, _ = masked_logit_scores.max(dim=1)
            max_other_softmax, _ = masked_softmax_scores.max(dim=1)

            batch_logit_margins = correct_logit_scores - max_other_logit  # [batch_size]
            batch_softmax_margins = correct_softmax_scores - max_other_softmax
            logit_margins.extend(batch_logit_margins.cpu().tolist())
            softmax_margins.extend(batch_softmax_margins.cpu().tolist())
            all_logit_confidences.extend(correct_logit_scores.cpu().tolist())
            all_softmax_confidences.extend(correct_softmax_scores.cpu().tolist())

    return all_logit_confidences, all_softmax_confidences, logit_margins, softmax_margins
