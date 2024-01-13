import torch
from torch import nn
from torch.nn.functional import relu, softmax
import numpy as np
from tqdm import tqdm

# Defining the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First convolutional layer (Input: 3 channels, Output: 8 feature maps, Kernel: 3x3)
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        # First pooling layer (Pooling: 2x2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional layer (Input: 8 feature maps, Output: 16 feature maps, Kernel: 3x3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        # Second pooling layer (Pooling: 2x2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third convolutional layer (Input: 16 feature maps, Output: 32 feature maps, Kernel: 3x3)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Third pooling layer (Pooling: 2x2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flattening the feature maps into a feature vector
        self.fc1 = nn.Linear(32 * 4 * 4, 128)

        # Linear layers / Fully connected layers
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # Output layer, assuming 10 classes for classification

    def forward(self, x):
        # Convolutional and pooling layers
        x = self.pool1(relu(self.conv1(x)))
        x = self.pool2(relu(self.conv2(x)))
        x = self.pool3(relu(self.conv3(x)))

        # Flattening the output for the dense layer
        x = x.view(-1, 32 * 4 * 4)

        # Fully connected layers
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        return x
