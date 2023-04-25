from torch.utils.data import Dataset, SubsetRandomSampler
import pandas as pd
import torch
import numpy as np

class point_dataset(Dataset):
    """
    Custom dataset based on our own data.
    There is no distinction between train data and test data yet.
    """
    def __init__(self, label_file:str):
        """
        Args:
            label_file(string): Path of csvfile.
        """
        self.df = pd.read_csv(label_file)
        pass

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx:int):
        """
        Args:
            idx(int): Index of item.

        Returns:
            (dict): 
                'data'(ndarray(float)): points data;
                'seq_len'(int): length of sequence;
                'label'(int): label class (0-3);
                'label_values'(ndarray(float)): label parameters' values.
        """
        points = self.df.iloc[idx, 0:200]
        seq_len = len(points)
        label_type = self.df.iloc[idx, -1]
        label_values = self.df.iloc[idx, 200:204]
        points = torch.from_numpy(points.to_numpy().astype(np.float32))
        points = points.unsqueeze(0)
        label_values = torch.from_numpy(label_values.to_numpy().astype(np.float32))
        sample = {'data': points, 'label': label_type, 'label_values': label_values, 'seq_len': seq_len}
        return sample
    

def dataset_random_split(dataset_size, test_split_rate):
    """
    Args:
        dataset_size(int): the number of datas in dataset.
        test_split_rate(float, (0,1)): the ratio of test dataset from the whole dataset.
    Returns:
        train_sampler, test_sampler
    """
    indices = list(range(dataset_size))
    split = int(np.floor(test_split_rate * dataset_size))
    np.random.shuffle(indices)
    return SubsetRandomSampler(indices[split:]), SubsetRandomSampler(indices[:split])