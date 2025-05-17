from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    learning_rate: float = 0.001
    batch_size: int = 128
    epochs_task1: int = 5
    epochs_task2: int = 5
    temperature: float = 2.0  # For LwF knowledge distillation
    lambda_old: float = 1.0  # Weight for old task loss in LwF
    seed: int = 42
    save_dir: str = "model_checkpoints"
    use_cuda: bool = True
    log_interval: int = 100
