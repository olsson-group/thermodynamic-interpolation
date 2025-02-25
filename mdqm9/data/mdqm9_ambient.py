import os
import sys
sys.path.append('../')

from rdkit import Chem
import numpy as np
import torch

from torch.utils import data
import torch_geometric

from thermo import utils

SCALING_FACTOR = 0.20754094  # in general across MDQM9 dataset
SCALING_FACTOR_31 = 0.09729941375   # small molecule
SCALING_FACTOR_10506 = 0.13163184188306332  # large molecule


class MDQM9MultiTempDataset(data.Dataset):

    """## MDQM9 dataset with data at multiple temperatures.
    """

    def __init__(self, traj_filename: str, sdf_filename: str, traj_path: str, sdf_path: str, split: str, 
                 Ts: list=[300], scale: bool=False, cutoff: float=np.inf) -> None:
        """## Initialize the dataset.

        ### Args:
            - `traj_filename (str)`: name of the trajectory.
            - `sdf_filename (str)`: name of the sdf file.
            - `traj_path (str)`: path to the trajectory file.
            - `sdf_path (str)`: path to the sdf file.
            - `split (str)`: which data split to load (e.g. train/test/val).
            - `Ts (list, optional)`: list of temperatures to load. Defaults to [300].
            - `scale (bool, optional)`: if True, the data is scaled before being processed. Defaults to False.
            - `cutoff (float, optional)`: cutoff for radius graph. Defaults to inf.

        ### Asserts:
            - `split in {'train', 'val', 'test'}`
        """

        assert split in {'train', 'val', 'test'}
        self.split = split

        # Note: temperature is in [300, 400, 500, 600, 700, 800, 900, 1000]
        self.temp_index_dict = dict(zip(np.arange(300, 1001, step=100), list(range(8))))   # map temperature to index

        self.data = [get_mdqm9_trajs(self.temp_index_dict[T], traj_filename, traj_path, scale=scale, split=split) for T in Ts]
        self.Ts = np.array([T for T in Ts for _ in range(len(self.data[0]))])

        self.data = np.concatenate(self.data, axis=0)

        self.atom_numbers = get_mdqm9_atom_numbers(traj_path, sdf_path, traj_filename, sdf_filename, distinguish=True, split=split)
        self.bond_index, self.bonds = get_bond_index_and_bonds(traj_filename, sdf_path, sdf_filename)

        self.add_radius_graph = utils.AddRadiusGraph(cutoff=cutoff)
        self.add_bond_graph = utils.AddBondGraph()
        self.coalesce = utils.Coalesce()

        super().__init__()
    
    def __len__(self) -> int:
        """## Get length of dataset.

        ### Returns:
            - `int`: The length of the dataset.
        """
        
        length, _, _ = self.data.shape
        return length
    
    def __getitem__(self, idx: int) -> dict:
        """## Get item from dataset.

        ### Args:
            - `idx (int)`: index of the item to retrieve.

        ### Returns:
            - `dict`: The item at the given index.
        """

        x = torch.tensor(self.data[idx], dtype=torch.float32).squeeze()
        T = torch.tensor(self.Ts[idx]).repeat(x.shape[0])

        return self.process(x, T)
    
    def process(self, x: torch.tensor, T: torch.tensor) -> torch_geometric.data.Batch:
        """## Package into a batch and Add edges.

        ### Args:
            - `x (torch.tensor)`: 3D cartesian coordinates of atoms in the molecule.
            - `T (torch.tensor)`: temperature of the molecule.

        ### Returns:
            - `torch_geometric.data.Batch`: The processed batch.
        """

        x = x - torch.mean(x, dim=0)    # remove center of mass to keep translation equivariance

        datalist = [torch_geometric.data.Data(x=x, T=T, atoms=self.atom_numbers, bond_index=self.bond_index, bonds=self.bonds)]
        batch = torch_geometric.data.Batch.from_data_list(datalist)

        batch = self.add_radius_graph(batch)
        batch = self.add_bond_graph(batch)
        batch = self.coalesce(batch)

        return batch


class MDQM9SamplerDataset(data.Dataset):
    """## Lightweight MDQM9 dataset for sampling.
    """
    
    def __init__(self, traj_filename: str, sdf_filename: str, traj_path: str, sdf_path: str, split: str='test', 
                 T0: int=300, T1: int=400, scale: bool=False, cutoff: float=np.inf, use_latent_trajs: bool=False, 
                 n_latent_samples: int=10_000, latent_traj_path="") -> None:

        assert split in {'train', 'val', 'test'}

        if use_latent_trajs:
            assert latent_traj_path != "", "latent_traj_path must be provided if use_latent_trajs is True"

        self.split = split

        if use_latent_trajs:
            self.data0, self.data, self.dlogp0 = get_latent_mdqm9_trajs(n_latent_samples, T0, scale=scale, traj_filename=traj_filename, traj_path=latent_traj_path)
        else:
            # Note: temperature is in [300, 400, 500, 600, 700, 800, 900, 1000]
            self.temp_index_dict = dict(zip(np.arange(300, 1001, step=100), list(range(8))))   # map temperature to index
            self.data = get_mdqm9_trajs(self.temp_index_dict[T0], traj_filename, traj_path, scale=scale, split=split)
            self.data0 = np.zeros_like(self.data)    # dummy data
            self.dlogp0 = np.zeros(len(self.data))
        
        self.T0s = np.ones(len(self.data)) * T0
        self.T1s = np.ones(len(self.data)) * T1

        self.atom_numbers = get_mdqm9_atom_numbers(traj_path, sdf_path, traj_filename, sdf_filename, distinguish=True, split=split)
        self.bond_index, self.bonds = get_bond_index_and_bonds(traj_filename, sdf_path, sdf_filename)

        self.add_radius_graph = utils.AddRadiusGraph(cutoff=cutoff)
        self.add_bond_graph = utils.AddBondGraph()
        self.coalesce = utils.Coalesce()

        super().__init__()
    
    def __len__(self) -> int:
        length, _, _ = self.data.shape
        return length
    
    def __getitem__(self, idx: int) -> torch_geometric.data.Batch:
        x = torch.tensor(self.data[idx], dtype=torch.float32).squeeze()
        z = torch.tensor(self.data0[idx], dtype=torch.float32).squeeze()
        dlogp0 = torch.tensor(self.dlogp0[idx], dtype=torch.float32).squeeze()

        T0 = torch.tensor(self.T0s[idx], dtype=torch.float32).repeat(x.shape[0])
        T1 = torch.tensor(self.T1s[idx], dtype=torch.float32).repeat(x.shape[0])

        return self.process(x, z, dlogp0, T0, T1)
    
    def process(self, x: torch.tensor, z: torch.tensor, dlogp0: torch.tensor, T0: torch.tensor, T1: torch.tensor) -> torch_geometric.data.Batch:
        x = x - torch.mean(x, dim=0)    # remove center of mass to keep translation equivariance
        z = z - torch.mean(z, dim=0)

        datalist = [torch_geometric.data.Data(latent_z=z, latent_dlogp=dlogp0, x=x, x0=x, T0=T0, T1=T1, atoms=self.atom_numbers, bond_index=self.bond_index, bonds=self.bonds)]
        batch = torch_geometric.data.Batch.from_data_list(datalist)

        batch = self.add_radius_graph(batch)
        batch = self.add_bond_graph(batch)
        batch = self.coalesce(batch)
        return batch


def get_latent_mdqm9_trajs(n_samples, T, scale: bool, traj_filename="00031.npy", traj_path: str="") -> np.array:
    assert traj_filename in {"00031.npy", "10506.npy"}
    traj_index = traj_filename.split(".")
    traj_index.pop()
    traj_index = "".join(traj_index)

    initial_traj = np.load(os.path.join(traj_path, f"samples_mol_{traj_index}_{T}k_forward.npy"))[:n_samples, 0, :, :]
    traj = np.load(os.path.join(traj_path, f"samples_mol_{traj_index}_{T}k_forward.npy"))[:n_samples, -1, :, :]
    dlogp0 = np.load(os.path.join(traj_path, f"dlogps_mol_{traj_index}_{T}k_forward.npy"))[:n_samples] if traj_filename == "00031.npy" else np.zeros(traj.shape[0])
    
    initial_coms = np.mean(initial_traj, axis=1)
    initial_traj = initial_traj - initial_coms[:, np.newaxis, :]

    coms = np.mean(traj, axis=1)
    traj = traj - coms[:, np.newaxis, :]

    # OBS: trajs are already scaled by 1/0.20754094 when loaded
    if not scale:
        traj = traj/SCALING_FACTOR #_31 if traj_filename == "00031.npy" else traj/SCALING_FACTOR_10506

        initial_traj = np.array(initial_traj)
        traj = np.array(traj)
    else:
        initial_traj = np.array(initial_traj)
        traj = np.array(traj)

    return initial_traj, traj, dlogp0


def get_mdqm9_trajs(temp_index, traj_filename, traj_path: str, scale: bool, split: str) -> np.array:
    trajs = np.load(os.path.join(traj_path, split, traj_filename))[temp_index, :, :, :]

    # calculate center of mass and center the data
    coms = np.mean(trajs, axis=1)
    trajs = trajs - coms[:, np.newaxis, :]

    if scale:
        trajs = trajs * SCALING_FACTOR_31 if traj_filename == "00031.npy" else trajs * SCALING_FACTOR_10506
        trajs = np.array(trajs)
    return trajs


def get_mdqm9_atom_numbers(traj_path, sdf_path, traj_filename, sdf_filename, distinguish: bool, split: str) -> torch.tensor:
    traj = np.load(os.path.join(traj_path, split, traj_filename))
    _, _, n_atoms, _ = traj.shape

    if distinguish:
        atom_numbers = np.arange(n_atoms)
    else:
        suppl = Chem.SDMolSupplier(os.path.join(sdf_path, sdf_filename), removeHs=False, sanitize=True)
        file_id = int(traj_filename.split('.')[0])

        mol = suppl[file_id]
        atom_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        atom_numbers = np.array(atom_numbers)
    return torch.tensor(atom_numbers, dtype=torch.long)


def get_bond_index_and_bonds(traj_filename: str, sdf_path: str, sdf_filename: str) -> tuple:
    suppl = Chem.SDMolSupplier(os.path.join(sdf_path, sdf_filename), removeHs=False, sanitize=True)
    file_id = int(traj_filename.split('.')[0])
    mol = suppl[file_id]

    bond_matrix = torch.tensor(
        [
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondTypeAsDouble())
            for bond in mol.GetBonds()
        ],
        dtype=torch.long,
    )
    bonds = bond_matrix[:, 2]
    edges = bond_matrix[:, :2].t().contiguous()
    src = torch.cat([edges[0], edges[1]], dim=0)
    dst = torch.cat([edges[1], edges[0]], dim=0)
    bond_index = torch.stack([src, dst], dim=0)

    bonds = torch.cat((bonds, bonds))
    return bond_index, bonds


