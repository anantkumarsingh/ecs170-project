import torch
from torch.utils.data import DataLoader

from scripts.Dataset_Loader_S2 import Stage2_Dataset
from scripts.Method_MLP_S2 import Method_MLP
from scripts.Setting_Train_Test_S2 import Setting_Train_Test
from scripts.Evaluate_Classification import Evaluate_Classification


def main():

    train_path = "../data/stage_2_data/train.csv"
    test_path = "../data/stage_2_data/test.csv"

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    # Load datasets
    train_dataset = Stage2_Dataset(train_path)
    test_dataset = Stage2_Dataset(test_path)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )

    # TESTING train_loader
    # for x_batch, y_batch in train_loader:
    #     print("X batch shape:", x_batch.shape)
    #     print("Y batch shape:", y_batch.shape)
    #     break


    model = Method_MLP()

    trainer = Setting_Train_Test(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device
    )

    trainer.train(epochs=12)
    # If I increase epochs too much, training loss may keep decreasing,
    # but test accuracy may stop improving or decrease.
    # That is a sign of overfitting.

    print("Best Test Accuracy During Training:", trainer.best_acc)

    # Earlier without the best_model_state, even if
    # the model performed better in some epochs,
    # the final accuracy and model state (weights) remained same
    # as the weights for last epoch.
    # Now, best_model_state retains the efficient weights.

    evaluator = Evaluate_Classification(
        model=model,
        test_loader=test_loader,
        device=device
    )

    results = evaluator.evaluate()
    print("Accuracy:", results["accuracy"])
    print("Macro Precision:", results["macro_precision"])
    print("Macro Recall:", results["macro_recall"])
    print("Macro F1:", results["macro_f1"])
    print("Weighted F1:", results["weighted_f1"])
    print("Confusion Matrix:")
    print(results["confusion_matrix"])

    trainer.plot_loss()

if __name__ == "__main__":
    main()