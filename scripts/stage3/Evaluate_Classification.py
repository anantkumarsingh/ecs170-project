import torch

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


class Evaluate_Classification:
    """
    Evaluates trained CNN model on test data
    """

    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device

    def evaluate(self):
        self.model.eval()

        all_preds = [] # prediction labels are stored here
        all_labels = []  # original labels are stored here

        with torch.no_grad(): # no training only testing, so it won't store or compute gradient values
            for x_batch, y_batch in self.test_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(x_batch)

                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        results = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "macro_precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
            "macro_recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
            "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
            "weighted_f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(all_labels, all_preds)

        }

        return results