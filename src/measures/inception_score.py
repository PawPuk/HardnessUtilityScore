from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.transforms import functional as TF
from tqdm import tqdm

from src.config.config import DEVICE


def compute_inception_score_using_inceptionV3(
    dataloader: torch.utils.data.DataLoader,
    splits: int = 10,
) -> Tuple[float, float]:
    """
    Compute Inception Score.

    Args:
        dataloader: yields batches where first element is images.
        splits: number of splits for mean/std.
    """
    # Load Inception v3
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    model = model.to(DEVICE)
    model.eval()

    # Precompute normalization constants
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

    all_probs = []

    with torch.no_grad():
        for images, _, _ in tqdm(dataloader, desc="Inception Score"):
            # Resize to 299x299 (bilinear)
            images = TF.resize(images, (299, 299))
            # Normalize
            images = images.to(DEVICE)
            images = (images - mean) / std

            logits = model(images)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)  # (N, 1000)

    # Compute IS over splits (sequential, as per original implementation)
    N = probs.shape[0]
    if N < splits:
        raise ValueError(f"Number of samples ({N}) < splits ({splits}).")

    split_size = N // splits
    scores = []

    for i in range(splits):
        start = i * split_size
        end = N if i == splits - 1 else (i + 1) * split_size
        p_y_given_x = probs[start:end]
        p_y = p_y_given_x.mean(axis=0, keepdims=True)
        kl = p_y_given_x * (np.log(p_y_given_x + 1e-12) - np.log(p_y + 1e-12))
        kl_mean = np.mean(kl.sum(axis=1))
        scores.append(np.exp(kl_mean))

    return float(np.mean(scores)), float(np.std(scores))
