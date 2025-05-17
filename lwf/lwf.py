import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from lwf.config import TrainingConfig
from lwf.model import SimpleConvNet
from lwf.trainer import (
    evaluate,
    get_dataset,
    lwf_train_epoch,
    save_model,
    standard_train_epoch,
)


class LwF:
    """
    Learning without Forgetting (LwF) implementation.
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize the LwF model.

        Args:
            config: Training configuration parameters
        """
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and config.use_cuda else "cpu"
        )

        # Set random seeds for reproducibility
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)

        # Create models for both approaches
        self.model_lwf = SimpleConvNet().to(self.device)
        self.model_standard = SimpleConvNet().to(self.device)

        # Optimizers
        self.optimizer_lwf = optim.Adam(
            self.model_lwf.parameters(), lr=config.learning_rate
        )
        self.optimizer_standard = optim.Adam(
            self.model_standard.parameters(), lr=config.learning_rate
        )

        # Keep track of old task model for LwF
        self.old_model = None

        # Store metrics for analysis
        self.metrics = {
            "lwf": {"task1": [], "task2": [], "task1_after_task2": []},
            "standard": {"task1": [], "task2": [], "task1_after_task2": []},
        }

    def train_task1(self, dataset_name: str = "mnist") -> None:
        """
        Train both models on the first task (MNIST).

        Args:
            dataset_name: Name of the dataset for Task 1 (default: 'mnist')
        """
        print(f"\n{'=' * 50}")
        print(f"Training on Task 1: {dataset_name.upper()}")
        print(f"{'=' * 50}")

        # Get dataset
        train_loader, test_loader = get_dataset(self.config, dataset_name)

        # Train standard model
        print("\nTraining standard model:")
        for epoch in range(1, self.config.epochs_task1 + 1):
            standard_train_epoch(
                self.config,
                self.model_standard,
                train_loader,
                self.optimizer_standard,
                epoch,
                self.device,
            )

            # Evaluate
            test_loss, accuracy = evaluate(
                self.model_standard, test_loader, self.device
            )
            print(
                f"Standard Model - Epoch {epoch}: Test loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%"
            )
            self.metrics["standard"]["task1"].append((epoch, test_loss, accuracy))

        # Train LwF model (for first task, this is same as standard training)
        print("\nTraining LwF model:")
        for epoch in range(1, self.config.epochs_task1 + 1):
            lwf_train_epoch(
                self.config,
                self.model_lwf,
                None,  # No old model for first task
                train_loader,
                self.optimizer_lwf,
                epoch,
                is_first_task=True,
                device=self.device,
            )

            # Evaluate
            test_loss, accuracy = evaluate(self.model_lwf, test_loader, self.device)
            print(
                f"LwF Model - Epoch {epoch}: Test loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%"
            )
            self.metrics["lwf"]["task1"].append((epoch, test_loss, accuracy))

        # Save task1 models
        save_model(self.model_standard, self.config.save_dir, "standard_task1")
        save_model(self.model_lwf, self.config.save_dir, "lwf_task1")

        # Create a copy of LwF model to use for knowledge distillation
        self.old_model = SimpleConvNet().to(self.device)
        self.old_model.load_state_dict(self.model_lwf.state_dict())
        self.old_model.eval()  # Set to evaluation mode

        # Final evaluation on task1
        test_loss, accuracy = evaluate(self.model_standard, test_loader, self.device)
        print(
            f"\nStandard Model - Final Task 1 Performance: Test loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )

        test_loss, accuracy = evaluate(self.model_lwf, test_loader, self.device)
        print(
            f"LwF Model - Final Task 1 Performance: Test loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )

    def train_task2(self, dataset_name: str = "fashion-mnist") -> None:
        """
        Train both models on the second task (Fashion-MNIST).

        Args:
            dataset_name: Name of the dataset for Task 2 (default: 'fashion-mnist')
        """
        print(f"\n{'=' * 50}")
        print(f"Training on Task 2: {dataset_name.upper()}")
        print(f"{'=' * 50}")

        # Get dataset for task 2
        train_loader, test_loader = get_dataset(self.config, dataset_name)

        # Also get test loader for task 1 to measure forgetting
        task1_test_loader = get_dataset(self.config, "mnist")[1]

        # Train standard model on task 2
        print("\nTraining standard model:")
        for epoch in range(1, self.config.epochs_task2 + 1):
            standard_train_epoch(
                self.config,
                self.model_standard,
                train_loader,
                self.optimizer_standard,
                epoch,
                self.device,
            )

            # Evaluate on task 2
            test_loss, accuracy = evaluate(
                self.model_standard, test_loader, self.device
            )
            print(f"Standard Model - Epoch {epoch}: Task 2 Accuracy: {accuracy:.2f}%")
            self.metrics["standard"]["task2"].append((epoch, test_loss, accuracy))

            # Evaluate on task 1 to measure forgetting
            task1_test_loss, task1_accuracy = evaluate(
                self.model_standard, task1_test_loader, self.device
            )
            print(
                f"Standard Model - Epoch {epoch}: Task 1 Accuracy (forgetting): {task1_accuracy:.2f}%"
            )
            self.metrics["standard"]["task1_after_task2"].append(
                (epoch, task1_test_loss, task1_accuracy)
            )

        # Train LwF model on task 2 with knowledge distillation
        print("\nTraining LwF model:")
        for epoch in range(1, self.config.epochs_task2 + 1):
            lwf_train_epoch(
                self.config,
                self.model_lwf,
                self.old_model,  # Use old model for knowledge distillation
                train_loader,
                self.optimizer_lwf,
                epoch,
                is_first_task=False,
                device=self.device,
            )

            # Evaluate on task 2
            test_loss, accuracy = evaluate(self.model_lwf, test_loader, self.device)
            print(f"LwF Model - Epoch {epoch}: Task 2 Accuracy: {accuracy:.2f}%")
            self.metrics["lwf"]["task2"].append((epoch, test_loss, accuracy))

            # Evaluate on task 1 to measure forgetting
            task1_test_loss, task1_accuracy = evaluate(
                self.model_lwf, task1_test_loader, self.device
            )
            print(
                f"LwF Model - Epoch {epoch}: Task 1 Accuracy (forgetting): {task1_accuracy:.2f}%"
            )
            self.metrics["lwf"]["task1_after_task2"].append(
                (epoch, task1_test_loss, task1_accuracy)
            )

        # Save task2 models
        save_model(self.model_standard, self.config.save_dir, "standard_task2")
        save_model(self.model_lwf, self.config.save_dir, "lwf_task2")

        # Final evaluations
        test_loss, accuracy = evaluate(self.model_standard, test_loader, self.device)
        print(
            f"\nStandard Model - Final Task 2 Performance: Test loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )

        test_loss, task1_accuracy = evaluate(
            self.model_standard, task1_test_loader, self.device
        )
        print(
            f"Standard Model - Final Task 1 Performance: Test loss: {test_loss:.4f}, Accuracy: {task1_accuracy:.2f}%"
        )

        test_loss, accuracy = evaluate(self.model_lwf, test_loader, self.device)
        print(
            f"LwF Model - Final Task 2 Performance: Test loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )

        test_loss, task1_accuracy = evaluate(
            self.model_lwf, task1_test_loader, self.device
        )
        print(
            f"LwF Model - Final Task 1 Performance: Test loss: {test_loss:.4f}, Accuracy: {task1_accuracy:.2f}%"
        )

    def plot_results(self) -> None:
        """
        Plot performance metrics for both models on both tasks.
        Analyzes catastrophic forgetting and shows comparative performance.
        """
        plt.figure(figsize=(15, 10))

        # Plot Task 1 performance before and after Task 2 training
        plt.subplot(2, 2, 1)
        # Final performance on Task 1 after Task 1 training
        standard_task1_final = self.metrics["standard"]["task1"][-1][2]
        lwf_task1_final = self.metrics["lwf"]["task1"][-1][2]

        # Performance on Task 1 after Task 2 training (forgetting measurement)
        epochs_task2 = list(range(1, self.config.epochs_task2 + 1))
        standard_task1_after = [
            x[2] for x in self.metrics["standard"]["task1_after_task2"]
        ]
        lwf_task1_after = [x[2] for x in self.metrics["lwf"]["task1_after_task2"]]

        plt.plot(
            [0] + epochs_task2,
            [standard_task1_final] + standard_task1_after,
            "b-",
            label="Standard",
        )
        plt.plot(
            [0] + epochs_task2, [lwf_task1_final] + lwf_task1_after, "r-", label="LwF"
        )
        plt.title("Task 1 Performance During Task 2 Training")
        plt.xlabel("Task 2 Epochs")
        plt.ylabel("Task 1 Accuracy (%)")
        plt.legend()
        plt.grid(True)

        # Plot Task 2 learning curves
        plt.subplot(2, 2, 2)
        standard_task2 = [x[2] for x in self.metrics["standard"]["task2"]]
        lwf_task2 = [x[2] for x in self.metrics["lwf"]["task2"]]

        plt.plot(epochs_task2, standard_task2, "b-", label="Standard")
        plt.plot(epochs_task2, lwf_task2, "r-", label="LwF")
        plt.title("Task 2 Learning Performance")
        plt.xlabel("Epochs")
        plt.ylabel("Task 2 Accuracy (%)")
        plt.legend()
        plt.grid(True)

        # Plot forgetting metric (difference between Task 1 accuracy before and after Task 2)
        plt.subplot(2, 2, 3)
        standard_forgetting = [
            standard_task1_final - acc for acc in standard_task1_after
        ]
        lwf_forgetting = [lwf_task1_final - acc for acc in lwf_task1_after]

        plt.plot(epochs_task2, standard_forgetting, "b-", label="Standard")
        plt.plot(epochs_task2, lwf_forgetting, "r-", label="LwF")
        plt.title("Catastrophic Forgetting Measurement")
        plt.xlabel("Task 2 Epochs")
        plt.ylabel("Task 1 Performance Degradation (%)")
        plt.legend()
        plt.grid(True)

        # Plot final performance comparison
        plt.subplot(2, 2, 4)
        models = ["Standard", "LwF"]

        # Final Task 1 performance after Task 2 training
        task1_final = [standard_task1_after[-1], lwf_task1_after[-1]]

        # Final Task 2 performance
        task2_final = [standard_task2[-1], lwf_task2[-1]]

        x = np.arange(len(models))
        width = 0.35

        plt.bar(x - width / 2, task1_final, width, label="Task 1")
        plt.bar(x + width / 2, task2_final, width, label="Task 2")

        plt.xlabel("Model")
        plt.ylabel("Accuracy (%)")
        plt.title("Final Performance Comparison")
        plt.xticks(x, models)
        plt.legend()
        plt.grid(True, axis="y")

        plt.tight_layout()

        # Save the figure
        try:
            plt.savefig("lwf_results.png")
            print("Results saved to lwf_results.png")
        except Exception as e:
            print(f"Error saving plot: {str(e)}")

        plt.show()
