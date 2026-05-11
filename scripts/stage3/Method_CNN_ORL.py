import torch
import torch.nn as nn
import torch.nn.functional as F


class Method_CNN_ORL(nn.Module):
    """
    CNN Model for ORL Data

    Input:
        Image Matrix: 1 x 112 x 92

    Output:
        40 class scores/logits (labels are from 1-40, but data loader sends 0-39 because of crossEntropy)
    """

    # NO input and output size params as dimensions in this case DON'T change / Easier to Debug if Hardcoded
    def __init__(self):
        super(Method_CNN_ORL, self).__init__()

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

        # SECOND CONV Layer
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # THIRD CONV Layer
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # FOURTH CONV Layer
        self.conv4 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
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
            in_features=128 * 14 * 11,
            out_features=128
        )

        self.fc2 = nn.Linear(
            in_features=128,
            out_features=40
        )


    def forward(self, x):
        """
        FIRST RUN: Bad. Very High Loss: 3.4309 and Accuracy was 0.125
        Possible Reasons: Deep NN. Orl is tiny. 20 epochs not enough.
        Model is too deep for small dataset. Used SGD at first, trying ADAM next.

        # CONV 2d --> Relu --> MaxPool
        x = self.pool(F.relu(self.conv1(x))) # o/p == 32 x 56 x 46

        # CONV 2d --> Relu --> MaxPool
        x = self.pool(F.relu(self.conv2(x))) # o/p == 64 x 28 x 23

        # CONV 2d --> Relu --> MaxPool
        x = self.pool(F.relu(self.conv3(x)))  # o/p == 128 x 14 x 11

        # CONV 2d --> Relu --> MaxPool
        x = self.pool(F.relu(self.conv4(x)))  # o/p == 128 x 7 x 5

        # Flatten x to pass into fc1
        x = x.view(x.size(0), -1)

        # Flatten --> FC1
        x = F.relu(self.fc1(x))

        # Output Layer
        x = self.fc2(x)

        """

        # CONV 2d --> Relu --> MaxPool
        x = self.pool(F.relu(self.conv1(x)))  # o/p == 32 x 56 x 46

        # CONV 2d --> Relu --> MaxPool
        x = self.pool(F.relu(self.conv2(x)))  # o/p == 64 x 28 x 23

        # CONV 2d --> Relu --> MaxPool
        x = self.pool(F.relu(self.conv3(x)))  # o/p == 128 x 14 x 11

        # Flatten x to pass into fc1
        x = x.view(x.size(0), -1)

        # Flatten --> FC1
        x = F.relu(self.fc1(x))

        # Output Layer
        x = self.fc2(x)

        return x
