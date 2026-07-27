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


def compute_margins(
    model: ResNet18LowRes,
    data_loader: torch.utils.data.DataLoader,
) -> Dict[int, List[float]]:
    """
    Compute margin for each sample: logit(true_label) - max_{c != true_label} logit(c).
    Returns a dictionary mapping each class index to a list of margins for that class.
    """
    model.eval()
    margins_by_class = {}

    with torch.no_grad():
        for images, labels, _ in data_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)  # [batch_size, num_classes]

            # Logits for the true class
            correct_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)

            # Mask out the true class to get max among all others
            masked_logits = logits.clone()
            masked_logits.scatter_(1, labels.unsqueeze(1), -float('inf'))
            max_other, _ = masked_logits.max(dim=1)

            batch_margins = correct_logits - max_other  # [batch_size]

            # Append each margin to its class list
            for margin, label in zip(batch_margins.cpu().tolist(), labels.cpu().tolist()):
                margins_by_class.setdefault(int(label), []).append(margin)

    return margins_by_class
