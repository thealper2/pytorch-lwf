import torch
import typer

from lwf.config import TrainingConfig
from lwf.lwf import LwF


def main(
    learning_rate: float = typer.Option(0.001, help="Learning rate for optimizer"),
    batch_size: int = typer.Option(128, help="Batch size for training"),
    epochs_task1: int = typer.Option(5, help="Number of epochs for Task 1 (MNIST)"),
    epochs_task2: int = typer.Option(
        5, help="Number of epochs for Task 2 (Fashion-MNIST)"
    ),
    temperature: float = typer.Option(
        2.0, help="Temperature parameter for LwF knowledge distillation"
    ),
    lambda_old: float = typer.Option(1.0, help="Weight for old task loss in LwF"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    save_dir: str = typer.Option(
        "model_checkpoints", help="Directory to save model checkpoints"
    ),
    use_cuda: bool = typer.Option(True, help="Use GPU if available"),
    log_interval: int = typer.Option(
        100, help="How many batches to wait before logging training status"
    ),
) -> None:
    # Create configuration
    config = TrainingConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs_task1=epochs_task1,
        epochs_task2=epochs_task2,
        temperature=temperature,
        lambda_old=lambda_old,
        seed=seed,
        save_dir=save_dir,
        use_cuda=use_cuda,
        log_interval=log_interval,
    )

    print(
        f"Device: {torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')}"
    )
    print(f"Configuration: {config}")

    # Create LwF model
    lwf_model = LwF(config)

    # Train and evaluate on Task 1 (MNIST)
    lwf_model.train_task1(dataset_name="mnist")

    # Train and evaluate on Task 2 (Fashion-MNIST)
    lwf_model.train_task2(dataset_name="fashion-mnist")

    # Plot results
    lwf_model.plot_results()


if __name__ == "__main__":
    typer.run(main)
