import os
import json
import argparse

import numpy as np

import torch
from torch.utils.data import random_split, DataLoader, Dataset

from thermo import interpolants


def get_interpolated_batch(x0s, x1s, beta0s, beta1s, interpolant: interpolants.BaseInterpolant) -> tuple:
    
    ts = torch.rand(x0s.shape[0], 1).to(x0s.device).to(x0s.dtype)
    xts_plus, xts_minus, zs = interpolant.calc_antithetic_xts(ts, x0s, x1s)
    
    xts_plus = xts_plus.to(torch.float64).squeeze(0).to(x0s.device)
    xts_minus = xts_minus.to(torch.float64).squeeze(0).to(x0s.device)
    zs = zs.to(torch.float64).squeeze(0).to(x0s.device)

    return (xts_plus, xts_minus, ts, beta0s, beta1s, zs)


def get_loaders(dataset: Dataset, config: argparse.Namespace) -> DataLoader:
    train_sz = int(0.8*len(dataset))
    val_sz = int(0.1*len(dataset))
    test_sz = len(dataset) - train_sz - val_sz

    train, val, test = random_split(dataset=dataset,
                                    lengths=[train_sz, val_sz, test_sz],
                                    generator=torch.Generator().manual_seed(config.seed))
    
    train_loader = DataLoader(dataset=train,
                              batch_size=config.batch_size,
                              shuffle=True,
                              drop_last=True,
                              generator=torch.Generator().manual_seed(config.seed))

    val_loader = DataLoader(dataset=val,
                            batch_size=config.batch_size,
                            shuffle=True,
                            drop_last=True,
                            generator=torch.Generator().manual_seed(config.seed))

    test_loader = DataLoader(dataset=test,
                             batch_size=config.batch_size,
                             shuffle=True,
                             drop_last=True,
                             generator=torch.Generator().manual_seed(config.seed))
    return train_loader, val_loader, test_loader


def load_config(path: str, filename: str) -> argparse.Namespace:
    """
    Load configuration file
    :param path: path to configuration file
    :param filename: configuration file name
    :return: configuration file
    """
    settings = json.load(open(os.path.join(path, filename), 'r'))

    parser = argparse.ArgumentParser()
    
    for key, value in settings.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)
    return parser.parse_args()


def add_to_json(json_path, data):
    if not os.path.exists(json_path):
        with open(json_path, 'w') as f:
            json.dump({}, f)
        f.close()

    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    json_data.update(data)
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    f.close()


class BoltzmannDensity:
    def __init__(self, beta: float, a: float=4, b:float=0.5) -> None:
        """
        Boltzmann density for asymmetric double well potential
        :param beta: inverse temperature
        :param a: potential parameter
        :param b: potential parameter
        :return: None
        """

        self.beta = beta
        self.a = a
        self.b = b

    def asymmetric_double_well(self, x: np.array) -> np.array:
        """
        Asymmetric double well potential
        :param x: position
        :return: potential energy
        """
        return self.a*(x**2 - 1)**2 + self.b*x

    def get_partition_function(self) -> np.array:
        """
        Numerical integration to get partition function
        :return: partition function
        """
        x = np.linspace(-50, 50, 10_000)
        unnormed_density = np.exp(-self.beta*self.asymmetric_double_well(x))
        return np.trapz(unnormed_density, x)

    def get_p(self, x: np.array)-> np.array:
        """
        Probability density
        :param x: position
        :return: probability density
        """
        Z = self.get_partition_function()
        p = np.exp(-self.beta*self.asymmetric_double_well(x))/Z
        return p

    def get_logp(self, x: np.array)-> np.array:
        """
        Log probability density
        :param x: position
        :return: log probability density
        """
        return np.log(self.get_p(x))

