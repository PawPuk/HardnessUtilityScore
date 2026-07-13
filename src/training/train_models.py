"""Core module that allows for training ensembles of models as well as estimating hardness."""

import os
from typing import cast, Dict, List, Sized, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.config.config import DEVICE, get_config
from src.hardness.estimators import compute_AUM
from src.models.neural_networks import ResNet18LowRes
from src.utils.evaluation import evaluate_model
from src.utils.io import save_in_hoc_hardness_estimates
from src.utils.reproducibility import compute_current_seed, set_reproducibility
from src.utils.structures import get_latest_model_indices


class ModelTrainer:
    """Allows training ensembles of models as well as estimating hardness."""
    def __init__(
            self,
            training_set_size: int,
            training_loaders: List[DataLoader],
            test_loader: Union[DataLoader, None],
            dataset_name: str,
            resampling_ratio: float = 0.0,
            save_probe_models: bool = True,
            estimate_hardness: bool = False,
            for_experiment_1: bool = False
    ):
        """
        Initialize the ModelTrainer class with configuration specific to the dataset.

        :param training_set_size: Specified the size of the training set. This is only useful for experiment1.py.
        :param training_loaders: List of DataLoaders for the training datasets. For experiment1.py where only one
        dataset is used pass the DataLoader in a List.
        :param test_loader: DataLoader for the test dataset.
        :param dataset_name: The name of the dataset being used.
        :param resampling_ratio: Ratio used for resampling. 0.0 indicates training on baseline datasets where no
        resampling was applied.
        :param save_probe_models: Whether to save the probe models after a specified epoch (default: True). We use this
        later to verify if probe models can be used for post-hoc hardness estimation.
        :param estimate_hardness: Specify if the hardness should be saved and stored during training (default False).
        This flag is only raised when estimating in-hoc hardness and not raised otherwise to reduce compute.
        """
        self.training_set_size = training_set_size
        self.training_loaders = training_loaders
        self.test_loader = test_loader
        self.resampling_ratio = resampling_ratio
        self.dataset_name = dataset_name
        self.save_probe_models = save_probe_models
        self.estimate_hardness = estimate_hardness

        self.config = get_config(self.dataset_name)

        self.num_epochs = self.config['num_epochs']
        # For estimate_hardness_in_hoc.py we train single ensemble as there is only one dataset.
        if for_experiment_1:
            self.num_models_to_train_per_dataset = self.config['num_datasets'] * self.config['num_models_per_dataset']
            self.dataset_count = 1
        else:
            self.num_models_to_train_per_dataset = self.config['num_models_per_dataset']
            self.dataset_count = self.config['num_datasets']

        self.save_dir = os.path.join(self.config['save_dir'], resampling_ratio, dataset_name)
        os.makedirs(self.save_dir, exist_ok=True)

    def train_model(
            self,
            current_dataset_index: int,
            current_model_index: int,
            in_hoc_hardness_estimates: Dict[Tuple[int, int], List[float]]
    ):
        """Train a single model."""
        dataset_model_id = (current_dataset_index, current_model_index)
        seed = compute_current_seed(self.config, current_dataset_index, current_model_index)
        set_reproducibility(seed)

        model = ResNet18LowRes(num_classes=self.config['num_classes']).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.config['lr'], momentum=self.config['momentum'],
                              weight_decay=self.config['weight_decay'], nesterov=True)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.config['lr_decay_milestones'], gamma=0.2)

        if self.estimate_hardness:
            # hardness_estimates[dataset_model_id][epoch_index][sample_index]: float
            in_hoc_hardness_estimates[dataset_model_id] = [[0.0 for _ in range(self.num_epochs)]
                                                           for _ in range(self.training_set_size)]

        for epoch in range(self.config['num_epochs']):
            model.train()
            running_loss, correct_train, total_train = 0.0, 0, 0

            for inputs, labels, indices in self.training_loaders[current_dataset_index]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += predicted.eq(labels).sum().item()

                if self.estimate_hardness:
                    compute_AUM(indices, outputs, labels, in_hoc_hardness_estimates, epoch, dataset_model_id)
            scheduler.step()

            # Report progress (accuracy & loss on training & test sets)
            if self.test_loader is not None:
                avg_test_loss, test_accuracy = evaluate_model(model, criterion, self.test_loader)
                avg_training_loss = running_loss / total_train
                training_accuracy = 100 * correct_train / total_train
                print(f'Model {current_model_index}, '
                      f'Epoch [{epoch + 1}/{self.config["num_epochs"]}] '
                      f'Training Loss: {avg_training_loss:.4f}, Training Acc: {training_accuracy:.2f}%, '
                      f'Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')

            # We save probe models to later verify if they can be used for post-hoc hardness estimation.
            if epoch + 1 == self.config['save_epoch']:
                if self.save_probe_models:
                    save_path = os.path.join(self.save_dir, f'dataset_{current_dataset_index}_model_'
                                                            f'{current_model_index}'
                                                            f'_epoch_{epoch + 1}.pth')
                    # Move the model to CPU to use it on local machines without CUDA support later.
                    model.to('cpu')
                    torch.save(model.state_dict(), save_path)
                    model.to(DEVICE)

        # Save model after full training. It will be used later for post-hoc hardness estimation.
        final_save_path = os.path.join(self.save_dir, f'dataset_{current_dataset_index}'
                                                      f'_model_{current_model_index}'
                                                      f'_epoch_{self.config["num_epochs"]}.pth')
        # Move the model to CPU to enable use on local machines without CUDA support for post-hoc hardness estimation.
        model.to('cpu')
        torch.save(model.state_dict(), final_save_path)

    def train_ensemble(
            self
    ):
        """Train an ensemble of models."""

        latest_model_indices = get_latest_model_indices(self.save_dir, self.config['num_epochs'], self.dataset_count)

        print(f"Number of samples in the training loader: {len(cast(Sized, self.training_loaders[0].dataset))}")
        print(f"Number of samples in the test loader: {len(cast(Sized, self.test_loader.dataset))}")
        print('-'*20)

        for dataset_id in tqdm(range(self.dataset_count)):
            for model_id in tqdm(range(latest_model_indices[dataset_id] + 1, self.num_models_to_train_per_dataset)):
                in_hoc_hardness_estimates = {(dataset_id, model_id): []}
                self.train_model(dataset_id, model_id, in_hoc_hardness_estimates)
                if self.estimate_hardness:
                    # Final step of in-hoc hardness estimation - averaging over all training signals.
                    in_hoc_hardness_estimates[(dataset_id, model_id)] = np.mean(
                        in_hoc_hardness_estimates[(dataset_id, model_id)], axis=1)
                    save_in_hoc_hardness_estimates(in_hoc_hardness_estimates, (dataset_id, model_id), self.dataset_name)
