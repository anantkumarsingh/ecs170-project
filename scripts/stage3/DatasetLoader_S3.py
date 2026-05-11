import torch
import pickle
from torch.utils.data import Dataset


class Stage3_Dataset(Dataset):
    """
    Depending on the input, open the ORL, MNIST, or CIFAR data.

    ARGS:
        data_to_load: ORL, MNIST, CIFAR
        split: train, test

    """

    def __init__(self, data_to_load, split):

        if data_to_load not in ["ORL", "MNIST", "CIFAR"]:
            raise ValueError('data_to_load must be "ORL", "MNIST", or "CIFAR"')

        if split not in ["train", "test"]:
            raise ValueError('split must be "train" or "test"')

        self.data_to_load = data_to_load
        self.split = split

        f = open('../../data/stage_3_data/'+data_to_load, 'rb')
        data = pickle.load(f)
        f.close()
        self.data = data[split]  # SAME AS self.data = data['train'] or data['test']

    def __getitem__(self, index): # Since we now have the appropriate dataset (train or test), we use it to get image_matrix and image_label OF i-th index
        image_matrix = self.data[index]['image']
        image_label = self.data[index]['label']

        # Converting Data to Tensors
        image_matrix = torch.tensor(image_matrix, dtype = torch.float32)

        # Normalizing Image Matrix values (from 0-255 to 0-1)
        # If already normalized, do nothing
        if image_matrix.max() > 1.0:
            image_matrix = image_matrix / 255.0
        image_label = torch.tensor(image_label, dtype = torch.long)

        # CNN expects image shape: channels × height × width
        if self.data_to_load == "MNIST":
            # MNIST is 28 × 28, so add channel dimension
            image_matrix = image_matrix.unsqueeze(0) # makes the dimensions 1 x 28 x 28 for MNIST
        elif self.data_to_load == "ORL":
            # ORL is 112 × 92 × 3, but channels are identical grayscale
            # Use only one channel
            image_matrix = image_matrix[:, :, 0]
            image_matrix = image_matrix.unsqueeze(0)

            # ORL labels are 1-40, but CrossEntropyLoss expects 0-39
            image_label = image_label - 1
        elif self.data_to_load == "CIFAR":
            # CIFAR is 32 × 32 × 3
            # Convert to 3 × 32 × 32
            image_matrix = image_matrix.permute(2, 0, 1)

        # return x, y
        return image_matrix, image_label

    def __len__(self):
        return len(self.data)




