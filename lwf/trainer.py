import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from lwf.config import TrainingConfig
from lwf.model import SimpleConvNet


def get_dataset(
    config: TrainingConfig, dataset_name: str
) -> Tuple[DataLoader, DataLoader]:
    """
    Get the specified dataset and create data loaders.

    Args:
        config: Training configuration
        dataset_name: Name of the dataset ('mnist' or 'fashion-mnist')

    Returns:
        Tuple of (train_loader, test_loader)

    Raises:
        ValueError: If dataset_name is not recognized
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    try:
        if dataset_name.lower() == "mnist":
            train_dataset = datasets.MNIST(
                "data", train=True, download=True, transform=transform
            )
            test_dataset = datasets.MNIST("data", train=False, transform=transform)

        elif dataset_name.lower() == "fashion-mnist":
            train_dataset = datasets.FashionMNIST(
                "data", train=True, download=True, transform=transform
            )
            test_dataset = datasets.FashionMNIST(
                "data", train=False, transform=transform
            )

        else:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. Use 'mnist' or 'fashion-mnist'."
            )

    except Exception as e:
        import traceback

        print(f"Error loading dataset {dataset_name}: {str(e)}")
        traceback.format_exc()
        raise

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    return train_loader, test_loader


def standard_train_epoch(
    config: TrainingConfig,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device=torch.device,
) -> float:
    """
    Train for one epoch using standard training approach.

    Args:
        config: Training configuration
        model: Neural network model
        train_loader: DataLoader for training data
        optimizer: Optimizer for model parameters
        epoch: Current epoch number
        device: Device to use for training

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    correct = 0
    processed = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        # Update progress bar
        if batch_idx % config.log_interval == 0:
            pbar.set_postfix(
                {"loss": loss.item(), "accuracy": 100.0 * correct / processed}
            )

    return total_loss / len(train_loader)


def lwf_train_epoch(
    config: TrainingConfig,
    model: nn.Module,
    old_model: Optional[nn.Module],
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    is_first_task: bool,
    device: torch.device,
) -> float:
    """
    Train for one epoch using LwF approach.

    Args:
        config: Training configuration
        model: Current model being trained
        old_model: Model from previous task (None for first task)
        train_loader: DataLoader for training data
        optimizer: Optimizer for model parameters
        epoch: Current epoch number
        is_first_task: Whether this is the first task (no distillation needed)
        device: Device to use for training

    Returns:
        Average loss for the epoch
    """
    model.train()
    if old_model is not None:
        old_model.eval()

    total_loss = 0
    correct = 0
    processed = 0

    pbar = tqdm(train_loader, desc=f"LwF Epoch {epoch}")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        # Standard cross-entropy loss for current task
        loss = F.cross_entropy(output, target)

        # Add knowledge distillation loss if not the first task
        if not is_first_task and old_model is not None:
            with torch.no_grad():
                old_output = old_model(data)

            # Apply temperature scaling
            soft_target = F.softmax(old_output / config.temperature, dim=1)
            output_scaled = output / config.temperature

            # Knowledge distillation loss (KL divergence)
            distillation_loss = F.kl_div(
                F.log_softmax(output_scaled, dim=1), soft_target, reduction="batchmean"
            ) * (config.temperature**2)

            # Total loss (weighted sum of cross-entropy and distillation loss)
            loss = loss + config.lambda_old * distillation_loss

        loss.backward()
        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        # Update progress bar
        if batch_idx % config.log_interval == 0:
            pbar.set_postfix(
                {"loss": loss.item(), "accuracy": 100.0 * correct / processed}
            )

    return total_loss / len(train_loader)


def evaluate(
    model: nn.Module, test_loader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate the model on test data.

    Args:
        model: Neural network model to evaluate
        test_loader: DataLoader for test data
        device: Device to use for evaluation

    Returns:
        Tuple of (test loss, accuracy)
    """
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction="sum").item()

            # Get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    return test_loss, accuracy


def save_model(model: nn.Module, save_dir: str, name: str) -> None:
    """
    Save model checkpoint.

    Args:
        model: Model to save
        save_dir: Directory to save the model
        name: Name of the checkpoint file
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = Path(save_dir) / f"{name}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
    except Exception as e:
        import traceback

        print(f"Error saving model: {str(e)}")
        print(traceback.format_exc())


def load_model(
    model: nn.Module, save_dir: str, name: str, device: torch.device
) -> None:
    """
    Load model from checkpoint.

    Args:
        model: Model to load weights into
        save_dir: Directory where the model is saved
        name: Name of the checkpoint file
        device: Device to load the model to

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
    """
    try:
        checkpoint_path = Path(save_dir) / f"{name}.pt"
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Model loaded from {checkpoint_path}")

    except FileNotFoundError:
        import traceback

        print(f"Checkpoint file {checkpoint_path} not found")
        print(traceback.format_exc())
        raise

    except Exception as e:
        import traceback

        print(f"Error loading model: {str(e)}")
        print(traceback.format_exc())
        raise
