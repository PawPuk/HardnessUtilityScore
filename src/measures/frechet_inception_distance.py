"""
Fréchet Inception Distance (FID) for evaluating generative models.
"""

import torch
import numpy as np
from torchvision.models import inception_v3, Inception_V3_Weights, Inception3
from torchvision.transforms import functional as TF
from scipy.linalg import sqrtm
from tqdm import tqdm

from src.config.config import DEVICE


def extract_features(mean: torch.Tensor, std: torch.Tensor,  model: Inception3,
                     loader: torch.utils.data.DataLoader, desc: str) -> np.ndarray:
    """Extract 2048-d features from all images in a DataLoader."""
    features = []
    with torch.no_grad():
        for images, _, _ in tqdm(loader, desc=desc):
            # Resize to 299x299 and normalize
            images = TF.resize(images, (299, 299))
            # Normalize
            images = images.to(DEVICE)
            images = (images - mean) / std

            feat = model(images)
            features.append(feat.cpu().numpy())
    return np.concatenate(features, axis=0)


def compute_fid(
    real_loader: torch.utils.data.DataLoader,
    gen_loader: torch.utils.data.DataLoader
) -> float:
    """
    Compute FID (data makes it class-conditioned but the code is general) between two datasets using Inception v3 features.

    Args:
        real_loader: DataLoader for real images (yields batches with first element as images).
        gen_loader: DataLoader for generated images (same format).

    Returns:
        FID score (float). Lower is better.
    """

    # Load Inception v3
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    # Remove the classification head to get 2048-dimensional features
    model.fc = torch.nn.Identity()
    model = model.to(DEVICE)
    model.eval()

    # Precompute normalization constants
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

    # Extract features
    features_real = extract_features(mean, std, model, real_loader, "Extracting real features")
    features_gen = extract_features(mean, std, model, gen_loader, "Extracting generated features")

    # Compute mean and covariance
    mu1 = np.mean(features_real, axis=0)
    mu2 = np.mean(features_gen, axis=0)
    sigma1 = np.cov(features_real, rowvar=False)
    sigma2 = np.cov(features_gen, rowvar=False)

    # Compute FID
    diff = mu1 - mu2
    mean_sq = np.dot(diff, diff)

    # Covariance term: trace(sigma1 + sigma2 - 2 * sqrt(sigma1 @ sigma2))
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = mean_sq + np.trace(sigma1 + sigma2 - 2 * covmean)

    return float(fid)
