import torch
import torch.nn as nn
import torch.optim as optim
import copy
import matplotlib.pyplot as plt
from sympy import print_tree

from scripts.stage3.DatasetLoader_S3 import Stage3_Dataset
from scripts.stage3.Method_CNN_MNIST import Method_CNN_MNIST
from scripts.stage3.Method_CNN_ORL import Method_CNN_ORL
from scripts.stage3.Method_CNN_CIFAR import Method_CNN_CIFAR

def get_model(dataset_name):

   """
    return correct CNN model based on dataset_name
        # MNIST -> Method_CNN_MNIST()
        # ORL -> Method_CNN_ORL()
        # CIFAR -> Method_CNN_CIFAR()

    """
   if dataset_name == "MNIST":
       return Method_CNN_MNIST()
   elif dataset_name == "ORL":
       return Method_CNN_ORL()
   elif dataset_name == "CIFAR":
       return Method_CNN_CIFAR()
   else:
       raise(ValueError, "Incorrect Dataset Name. Try: MNIST, ORL, or CIFAR")

class Setting_Train_Test_S3:
    """
    Handles model training for Stage 3.
    """

    def __init__(self, model, train_loader, test_loader, device, model_optimizer, learning_rate=0.01, momentum=0.9):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.best_acc = 0.0         # Initial Accuracy. Changes with each epoch.
        self.best_model_state = None        # Final Accuracy and Chosen Weights.

        # Loss function for multiclass classification
        self.criterion = nn.CrossEntropyLoss()

        """
        SGD works great for MNIST, we have 99% accuracy.
        We need different optimizer for other 2 datasets.
        This is passed from main script onto here.
        """
        # Optimizer updates model weights
        if model_optimizer == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=momentum
            )
        elif model_optimizer == "Adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate
            )
        else: raise (ValueError, "Try using 'SGD', or 'Adam'")

        print(f"Optimizer Chosen: {model_optimizer}")

        # Stores average training loss per epoch
        self.train_losses = []

    def train(self, epochs=10):

        self.model.to(self.device)

        for epoch in range(epochs):
            self.model.train()

            total_loss = 0.0

            for x_batch, y_batch in self.train_loader:

                # Move data to GPU/MPS/CPU
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Clear old gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(x_batch)

                # Compute loss
                loss = self.criterion(outputs, y_batch)

                # Backward pass
                loss.backward()

                # Update weights
                self.optimizer.step()

                # Add batch loss
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            test_acc = self.test_accuracy() # Testing accuracy side by side as epoch increases

            if test_acc > self.best_acc: # If new accuracy is better than current best accuracy
                self.best_acc = test_acc
                # Save the model’s weights at this moment
                self.best_model_state = copy.deepcopy(self.model.state_dict())

            print(
                f"Epoch [{epoch + 1}/{epochs}], "
                f"Loss: {avg_loss:.4f}, "
                f"Test Accuracy: {test_acc:.4f}"
            )
            self.train_losses.append(avg_loss)

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

    def test_accuracy(self):
        """
        Tests accuracy of model at each epoch.
        Helps store final high accuracy of model.
        """
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for x_batch, y_batch in self.test_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(x_batch)
                preds = torch.argmax(outputs, dim=1)

                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        return correct / total

    def plot_loss(self, dataset_name):

        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses)
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title("Training Loss Curve")

        # SAVE IMAGE
        plt.savefig(f"../{dataset_name}_training_loss.png")  # saves in project root directory

        plt.show()