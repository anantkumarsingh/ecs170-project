import pandas as pd
import torch
from torch.utils.data import Dataset


class Stage2_Dataset(Dataset):
    """
    Dataset for Stage 2.

    Each row format:
    label, feature1, feature2, ..., feature784

    TODO:
    - Read csv file
    - Separate labels and features
    - Normalize feature values (neural networks work better when inputs are around 0-1)
    - Return tensors for PyTorch
    """

    def __init__(self, file_path):
        self.file_path = file_path

        # TODO: load CSV using pandas
        self.data = pd.read_csv(file_path)

        # TODO: first column is label
        labels = self.data.iloc[: , 0].values # returns first column as 1-D object

        # TODO: remaining 784 columns are image features
        features = self.data.iloc[: , 1: ].values # returns all the other columns

        # TODO: normalize features from 0-255 to 0-1
        features = features / 255.0

        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        # return number of rows
        return self.data.shape[0]

    def __getitem__(self, index):
        # return x and y
        return self.features[index], self.labels[index]
