from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights, Inception3
from torchvision.transforms import functional as TF
from tqdm import tqdm

from src.config.config import DEVICE, get_config
from src.models.neural_networks import ResNet18LowRes
from src.utils.io import extract_paths_to_pretrained_models


def load_classifier(
    dataset_name: str,
    num_classes: int,
    model_type: str
) -> Union[Inception3, ResNet18LowRes]:
    """
    Load a classifier (with the final linear layer intact) for Inception Score.
    """
    if model_type == 'InceptionV3':
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)  # output: 1000 logits
    elif model_type == 'ResNet18LowRes':
        model_paths = extract_paths_to_pretrained_models(dataset_name)
        model = ResNet18LowRes(num_classes)
        # Load the pretrained weights (the model already has the correct fc layer)
        state_dict = torch.load(model_paths[0][0])
        model.load_state_dict(state_dict)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(DEVICE)
    model.eval()
    return model


def compute_inception_score(
    dataloader: torch.utils.data.DataLoader,
    dataset_name: str,
    model_type: str = 'InceptionV3',
    weird_normalization: bool = False,  # if True, use ImageNet stats; else use dataset stats
    splits: int = 10
) -> Tuple[float, float]:
    """
    Compute Inception Score using either InceptionV3 or ResNet18LowRes.

    Args:
        dataloader: yields batches where first element is images.
        dataset_name: name of the dataset (to fetch config for mean/std).
        model_type: 'InceptionV3' or 'ResNet18LowRes'.
        weird_normalization: if True, use ImageNet stats (0.485,0.456,0.406 / 0.229,0.224,0.225).
                             if False, use dataset stats from config.
        splits: number of splits for mean/std.
    """
    config = get_config(dataset_name)
    num_classes = config['num_classes']

    # Load classifier
    model = load_classifier(dataset_name, num_classes, model_type)

    # Choose normalization stats
    if weird_normalization:
        mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
    else:
        mean = torch.tensor(config['mean'], device=DEVICE).view(1, 3, 1, 1)
        std = torch.tensor(config['std'], device=DEVICE).view(1, 3, 1, 1)

    all_probs = []

    with torch.no_grad():
        for images, _, _ in tqdm(dataloader, desc=f"Inception Score ({model_type})"):
            # Resize for InceptionV3 only; ResNet expects 32x32
            if model_type == 'InceptionV3':
                images = TF.resize(images, (299, 299))

            images = images.to(DEVICE)
            images = (images - mean) / std

            logits = model(images)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)  # (N, 1000) or (N, num_classes) depending on model_type.

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
