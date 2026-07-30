from typing import Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import sqrtm
import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights, Inception3
from torchvision.transforms import functional as TF
from tqdm import tqdm

from src.config.config import DEVICE, get_config
from src.models.neural_networks import ResNet18LowRes
from src.utils.io import extract_paths_to_pretrained_models


def load_feature_extractor(
        dataset_name: str,
        num_classes: int,
        model_type: str
) -> Union[Inception3, ResNet18LowRes]:
    """Load a feature extractor (removes classification head)."""
    if model_type == 'InceptionV3':
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()   # output: 2048 dims
    elif model_type == 'ResNet18LowRes':
        model_paths = extract_paths_to_pretrained_models(dataset_name)
        model = ResNet18LowRes(num_classes)
        # First, load the full classifier to get the correct weights
        state_dict = torch.load(model_paths[0][0])
        model.load_state_dict(state_dict)
        # Now remove the classifier to get features
        model.fc = nn.Identity()   # output: 512 dims
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(DEVICE)
    model.eval()
    return model


def extract_features(
        loader: torch.utils.data.DataLoader,
        model: Union[Inception3, ResNet18LowRes],
        weird_normalization: bool,
        dataset_mean: Tuple[float, float, float],
        dataset_std: Tuple[float, float, float],
        desc: str = "Extracting features"

) -> NDArray:
    features = []
    with torch.no_grad():
        for images, _, _ in tqdm(loader, desc=desc):
            if isinstance(model, Inception3):
                images = TF.resize(images, (299, 299))
            if weird_normalization:
                mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
            else:
                mean = torch.tensor(dataset_mean, device=DEVICE).view(1, 3, 1, 1)
                std = torch.tensor(dataset_std, device=DEVICE).view(1, 3, 1, 1)
            images = (images - mean) / std
            images = images.to(DEVICE)

            feat = model(images)
            features.append(feat.cpu().numpy())
    return np.concatenate(features, axis=0)


def compute_fid(
        dataset_name: str,
        real_loader: torch.utils.data.DataLoader,
        gen_loader: torch.utils.data.DataLoader,
        model_type: str = 'InceptionV3',
        weird_normalization: bool = False
) -> float:
    config = get_config(dataset_name)
    num_classes = config['num_classes']
    mean = config['mean']
    std = config['std']

    # Load model
    model = load_feature_extractor(dataset_name, num_classes, model_type)

    # Extract features
    feat_real = extract_features(real_loader, model, weird_normalization, mean, std, "Real (FID)")
    feat_gen = extract_features(gen_loader, model, weird_normalization, mean, std, "Generated (FID)")

    mu1, sigma1 = np.mean(feat_real, axis=0), np.cov(feat_real, rowvar=False)
    mu2, sigma2 = np.mean(feat_gen, axis=0), np.cov(feat_gen, rowvar=False)

    # Compute FID
    diff = mu1 - mu2
    mean_sq = np.dot(diff, diff)

    # Covariance term: trace(sigma1 + sigma2 - 2 * sqrt(sigma1 @ sigma2))
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = mean_sq + np.trace(sigma1 + sigma2 - 2 * covmean)

    return float(fid)
