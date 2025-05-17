import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleConvNet(nn.Module):
    """A simple convolutional neural network for MNIST/Fashion-MNIST classification."""

    def __init__(self, num_classes: int = 10):
        """
        Initialize the network architecture.

        Args:
            num_classes: Number of output classes (default: 10 for MNIST/Fashion-MNIST)
        """
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x: Input tensor of shape [batch_size, 1, 28, 28]

        Returns:
            Output logits of shape [batch_size, batch_size]
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the network (up to the last hidden layer).

        Args:
            x: Input tensor of shape [batch_size, 1, 28, 28]

        Returns:
            Feature tensor of shape [batch_size, 128]
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        return x
