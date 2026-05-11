import torch
from sympy import print_tree
from torch.utils.data import DataLoader
import torch.optim as optim
from scripts.stage3.DatasetLoader_S3 import Stage3_Dataset
from scripts.stage3.Method_CNN_MNIST import Method_CNN_MNIST
from scripts.stage3.Method_CNN_ORL import Method_CNN_ORL
from scripts.stage3.Method_CNN_CIFAR import Method_CNN_CIFAR
from scripts.stage3.Setting_Train_Test_S3 import Setting_Train_Test_S3
from scripts.stage3.Evaluate_Classification import Evaluate_Classification


def get_device():
    """
    Check for GPU device and use for parallel processing. Faster execution.
    :return: device available
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def run_experiment(dataset_name, model, batch_size, epochs, learning_rate, momentum, device, model_optimizer):

    print(f"\nRunning experiment for {dataset_name}")
    train_dataset = Stage3_Dataset(dataset_name, split="train")
    test_dataset = Stage3_Dataset(dataset_name, split="test")

    # DEBUG: check one sample
    x, y = train_dataset[0]

    print("Image shape:", x.shape)
    print("Label:", y)
    print("Image min:", x.min().item())
    print("Image max:", x.max().item())
    print("Image mean:", x.mean().item())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    trainer = Setting_Train_Test_S3(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        model_optimizer=model_optimizer,
        learning_rate=learning_rate,
        momentum=momentum
    )

    # For DEBUG
    # Issue with ORL: Maybe data was already normalized. So normalization was not needed.
    # Fixed with adding an IF condition in DATASET LOADER for Image Matrix values
    # DEBUG LINES KEPT for better visualization of data

    labels = [train_dataset[i][1].item() for i in range(len(train_dataset))]

    print("Train size:", len(train_dataset))
    print("Min label:", min(labels))
    print("Max label:", max(labels))
    print("Unique labels:", sorted(set(labels)))
    print("Number of unique labels:", len(set(labels)))

    trainer.train(epochs=epochs)
    print(f"Best Test Accuracy for {dataset_name}: {trainer.best_acc:.4f}")
    trainer.plot_loss(dataset_name)

    evaluator = Evaluate_Classification(
        model=model,
        test_loader=test_loader,
        device=device
    )
    results = evaluator.evaluate()
    print(f"======= STATS FOR {dataset_name} =======")
    print("Accuracy:", results["accuracy"])
    print("Macro Precision:", results["macro_precision"])
    print("Macro Recall:", results["macro_recall"])
    print("Macro F1:", results["macro_f1"])
    print("Weighted F1:", results["weighted_f1"])
    print("Confusion Matrix:")
    print(results["confusion_matrix"])

    print()
    print("=" * 40)
    print()


def main():

    # Device setup
    device = get_device()
    print("Using device:", device)

    # MNIST was performing VERY GOOD first try.
    # Optimizer used is SGD. 99% accuracy first try.

    run_experiment(
        dataset_name="MNIST",
        model=Method_CNN_MNIST(),
        batch_size=64,
        epochs=10,
        learning_rate=0.01,
        momentum=0.9,
        device=device,
        model_optimizer="SGD"
    )

    # ORL was performing BAD first try.
    # Was using SINGLE optimizer (SGD) for all 3 of them which is not good.
    # FIX by adding a model_optimizer param. So we have diff optimizers for each dataset.
    # Adam runs surprisingly well for ORL
    # That "surprising" could be because there is slight MEMORIZATION due to small sample size
    # Earlier learning rate was 0.005 which is VERY HIGH for this and not good at all.
    # Switched to 0.001 which is standard for ADAM. Model performs much better
    # Accuracy now is around 92-97%

    run_experiment(
        dataset_name="ORL",
        model=Method_CNN_ORL(),
        batch_size=20,
        epochs=15,
        learning_rate=0.001,
        momentum=0.9, # NOT USED SINCE, ADAM
        device=device,
        model_optimizer="Adam"
    )

    # CIFAR was also below average first try.
    # Mainly because SGD was being used for all 3 datasets. About 63% accuracy.
    # Was automatically fixed when I fixed Normalization in Dataset Loader and separate optimizer accessibility
    # About 75% accuracy
    # Takes more time, so epochs are 15 instead of 20

    run_experiment(
        dataset_name="CIFAR",
        model=Method_CNN_CIFAR(),
        batch_size=64,
        epochs=15,
        learning_rate=0.001,
        momentum=0.9, # NOT USED SINCE, ADAM
        device=device,
        model_optimizer="Adam"
    )

if __name__ == "__main__":
    main()