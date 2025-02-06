import numpy as np
import pandas as pd

import torch
from torch.utils import data


class ADWMultiTempDataset(data.Dataset):

    """
    MDQM9 replica exchange dataset with data at multiple temperatures.
    """

    def __init__(self, n_samples, traj_filename: str="samples.csv", traj_path: str="dataset", betas: list=[2.0], scale: bool=False, seed=0) -> None:
        """
        Initialize the dataset.
        :param traj_filename: Name of the trajectory file.
        :param traj_path: Path to the trajectory data.
        :param betas: Initial inverse temperatures
        :param scale: If True, the data is scaled by the standard deviation.
        :return: None
        """

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.data = [get_adw_trajs(beta, traj_path, traj_filename, scale) for beta in betas]
        
        self.betas = np.array([beta for beta in betas for _ in range(len(self.data[0]))])
        self.data = np.concatenate(self.data, axis=0)

        indexes = np.arange(len(self.data))
        np.random.shuffle(indexes)
        indexes = indexes[:n_samples]

        self.data = self.data[indexes]
        self.betas = self.betas[indexes]

        super().__init__()
    
    def __len__(self) -> int:
        """
        Standard len method for PyTorch datasets.
        :return: The length of the dataset.
        """
        length = self.data.shape[0]
        return length
    
    def __getitem__(self, idx: int) -> dict:
        """
        Standard getitem method for PyTorch datasets.
        :param idx: Index of the item to retrieve.
        :return: The item at the given index.
        """

        x = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0)
        beta = torch.tensor(self.betas[idx]).unsqueeze(0)
        return (x, beta)
 

def get_adw_trajs(beta: float, path: str="dataset", filename: str="samples.csv", scale: bool=False):
    dataset = pd.read_csv(f"{path}/{filename}")[f'{beta:.2f}'].values

    if scale:
        dataset = (dataset - np.mean(dataset, axis=0)) / np.std(dataset, axis=0)
    return dataset


if __name__ == "__main__":
    dataset = ADWMultiTempDataset(traj_filename="samples.csv", traj_path="dataset", betas=[1.5, 2.0])
    print(dataset[0].beta)


