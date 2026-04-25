import torch
import torch.nn as nn


class Method_MLP(nn.Module):
    """
    MLP model for Stage 2 multiclass classification.

    Input:
        784 features

    Output:
        10 class scores/logits (labels are from 0-9)
    """

    """
    Uncomment the functions (and comment the other versions) 
    to test original model framework with 2 hidden layers
    """
    # def __init__(self, input_size=784, hidden_1=128, hidden_2=64, output_size=10): # choice of default hyperparameters 128, 64
    #     super(Method_MLP, self).__init__()
    #
    #     self.model = nn.Sequential(
    #         nn.Linear(input_size, hidden_1),
    #         nn.ReLU(),
    #
    #         nn.Linear(hidden_1, hidden_2),
    #         nn.ReLU(),
    #
    #         nn.Linear(hidden_2, output_size)
    #         # We do not apply Softmax here because nn.CrossEntropyLoss
    #         # internally applies LogSoftmax in a numerically stable way.
    #         # Adding Softmax manually would cause instability and incorrect gradients.
    #     )

    """
    Model with 3 hidden layers. Deeper architecture.
    """

    # def __init__(self, input_size=784, output_size=10): # choice of default hyperparameters 128, 64
    #     super(Method_MLP, self).__init__()
    #
    #     self.model = nn.Sequential(
    #         nn.Linear(input_size, 256),
    #         nn.ReLU(),
    #
    #         nn.Linear(256, 128),
    #         nn.ReLU(),
    #
    #         nn.Linear(128, 64),
    #         nn.ReLU(),
    #
    #         nn.Linear(64, output_size)
    #
    #     )

    """
    Model with 3 hidden layers. With Dropout.
    Current model. after experimenting with No. of Hidden Layers and Input size. Best Performance, yet.
    """

    def __init__(self, input_size=784, output_size=10): # choice of default hyperparameters 128, 64
        super(Method_MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, output_size)

        )



    def forward(self, x):
        return self.model(x)