import torch
import torch.nn as nn
import torch.nn.functional as F


class Method_CNN_MNIST(nn.Module):
    """
    CNN Model for MNIST Data

    Input:
        Image Matrix: 1 x 28 x 28

    Output:
        10 class scores/logits (labels are from 0-9)
    """

    # NO input and output size params as dimensions in this case DON'T change / Easier to Debug if Hardcoded
    def __init__(self):
        super(Method_CNN_MNIST, self).__init__()

        # NOT USING Sequential as it has limitations
        # We may need skip connections, branches, etc.

        # FIRST CONV Layer
        self.conv1 = nn.Conv2d(
            in_channels= 1,
            out_channels= 32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # FIRST CONV Layer
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Max Pooling Layer
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Fully Connected Layers
        self.fc1 = nn.Linear(
            in_features=64 * 7 * 7,
            out_features=128
        )

        self.fc2 = nn.Linear(
            in_features=128,
            out_features=10
        )


    def forward(self, x):
        # CONV 2d --> Relu --> MaxPool
        x = self.pool(F.relu(self.conv1(x))) # o/p == 32 x 14 x 14

        # CONV 2d --> Relu --> MaxPool
        x = self.pool(F.relu(self.conv2(x))) # o/p == 64 x 7 x 7

        # Flatten x to pass into fc1
        x = x.view(x.size(0), -1)

        # Flatten --> FC1
        x = F.relu(self.fc1(x))

        # Output Layer
        x = self.fc2(x)

        return x
