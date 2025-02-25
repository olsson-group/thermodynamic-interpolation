import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from rdkit import Chem
from scipy.spatial.transform import Rotation
import numpy as np
import torch

from torch.utils import data
import torch_geometric
from torch_geometric.loader import DataLoader

from mdqm9.thermo import utils

SCALING_FACTOR = 0.20754094  # in general across MDQM9 dataset
SCALING_FACTOR_31 = 0.09729941375   # small molecule
SCALING_FACTOR_10506 = 0.13163184188306332  # large molecule


class MDQM9MultiTempDataset(data.Dataset):
    
    """## MDQM9 dataset with data at one temperature.
    """

    def __init__(self, traj_filename: str, sdf_filename: str, traj_path: str, sdf_path: str, split: str, 
                 Ts: list[300, 400, 500], scale: bool=False, cutoff: float=np.inf, seed=0, align=True) -> None:
        
        """Initialize the dataset.

        Args:
            traj_filename (str): Name of the trajectory file.
            sdf_filename (str): Name of the SDF file.
            traj_path (str): Path to the trajectory file.
            sdf_path (str): Path to the SDF file.
            split (str): Train, validation, or test split.
            T (float): Temperature of the data.
            scale (bool): Whether to scale the data.
            cutoff (float): Cutoff distance for the neighbor list.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        assert split in ['train', 'val', 'test']
        self.split = split
        self.align = align

        # Note: temperature is in [300, 400, 500, 600, 700, 800, 900, 1000]
        self.temp_index_dict = dict(zip(np.arange(300, 1001, step=100), list(range(8))))   # map temperature to index
        self.data = [get_mdqm9_trajs(self.temp_index_dict[T], traj_filename, traj_path, scale, split=split) for T in Ts]
        self.Ts = np.array([T for T in Ts for _ in range(len(self.data[0]))])
        print(self.Ts.shape)
        self.data = np.concatenate(self.data, axis=0)

        self.atom_numbers = get_mdqm9_atom_numbers(traj_path, sdf_path, traj_filename, sdf_filename, distinguish=False, split=split)
        self.atom_order = get_mdqm9_atom_numbers(traj_path, sdf_path, traj_filename, sdf_filename, distinguish=True, split=split)
        self.bond_index, self.bonds = get_bond_index_and_bonds(traj_filename, sdf_path, sdf_filename)

        self.add_radius_graph = utils.AddRadiusGraph(cutoff=cutoff)
        self.add_bond_graph = utils.AddBondGraph()
        self.coalesce = utils.Coalesce()

        super().__init__()

    def __len__(self) -> int:
        """## Get length of dataset.

        ### Returns:
            - `int`: length of the dataset.
        """

        length, _, _ = self.data.shape
        return length
    
    def __getitem__(self, idx: int) -> dict:
        """## Get item from dataset.

        ### Args:
            - `idx (int)`: index of the item.
        
        ### Returns:
            - `torch_geometric.data.Batch`: batch item at the given index.
        """

        x1 = torch.tensor(self.data[idx], dtype=torch.float32).squeeze()
        x0 = torch.randn_like(x1)
        T = torch.tensor(self.Ts[idx].repeat(x1.shape[0]))

        return self.process(x0, x1, T)

    def process(self, x0: torch.Tensor, x1: torch.Tensor, T: torch.tensor) -> torch_geometric.data.Batch:
        """## Process the data.

        ### Args:
            - `x (torch.Tensor)`: input data.

        ### Returns:
            - `torch_geometric.data.Batch`: processed data.
        """

        x0 = x0 - torch.mean(x0, axis=0)
        x1 = x1 - torch.mean(x1, axis=0)

        if self.align:
            r = Rotation.align_vectors(a=x1.numpy(), b=x0.numpy(), weights=None)[0]  # create rotation of x0 to x1:s reference frame
            x0 = torch.tensor(r.apply(x0.numpy()), dtype=torch.float32)

        datalist = [torch_geometric.data.Data(x=torch.zeros_like(x0), x0=x0, x1=x1, T=T, atom_number=self.atom_order, bond_index=self.bond_index, bonds=self.bonds)]
        batch = torch_geometric.data.Batch.from_data_list(datalist)

        batch = self.add_radius_graph(batch)
        batch = self.add_bond_graph(batch)
        batch = self.coalesce(batch)
        return batch

class SamplerDataset(data.Dataset):
    """## Sampler dataset for bg generation.
    """

    def __init__(self, traj_filename: str, sdf_filename: str, traj_path: str, sdf_path: str, split: str, 
                 Ts: list[300, 400, 500], n_samples: int, scale: bool=False, cutoff: float=np.inf, seed=0, align=True) -> None:
        
        """Initialize the dataset.

        Args:
            traj_filename (str): Name of the trajectory file.
            sdf_filename (str): Name of the SDF file.
            traj_path (str): Path to the trajectory file.
            sdf_path (str): Path to the SDF file.
            split (str): Train, validation, or test split.
            T (float): Temperature of the data.
            scale (bool): Whether to scale the data.
            cutoff (float): Cutoff distance for the neighbor list.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        assert split in ['train', 'val', 'test']
        self.split = split
        self.align = align
        self.n = n_samples
        # Note: temperature is in [300, 400, 500, 600, 700, 800, 900, 1000]
        self.temp_index_dict = dict(zip(np.arange(300, 1001, step=100), list(range(8))))   # map temperature to index

        self.T = Ts[0]
        self.data = [get_mdqm9_trajs(self.temp_index_dict[self.T], traj_filename, traj_path, scale, split=split)]
        self.data = np.concatenate(self.data, axis=0)
        self.x1 = torch.tensor(self.data[0], dtype=torch.float32).squeeze()
        self.atom_numbers = get_mdqm9_atom_numbers(traj_path, sdf_path, traj_filename, sdf_filename, distinguish=False, split=split)
        self.atom_order = get_mdqm9_atom_numbers(traj_path, sdf_path, traj_filename, sdf_filename, distinguish=True, split=split)
        self.bond_index, self.bonds = get_bond_index_and_bonds(traj_filename, sdf_path, sdf_filename)

        self.add_radius_graph = utils.AddRadiusGraph(cutoff=cutoff)
        self.add_bond_graph = utils.AddBondGraph()
        self.coalesce = utils.Coalesce()

        super().__init__()

    def __len__(self) -> int:
        """## Get length of dataset.

        ### Returns:
            - `int`: length of the dataset.
        """

        length = self.n
        return length
    
    def __getitem__(self, idx: int) -> dict:
        """## Get item from dataset.

        ### Args:
            - `idx (int)`: index of the item.
        
        ### Returns:
            - `torch_geometric.data.Batch`: batch item at the given index.
        """

        # x1 = torch.tensor(self.data[idx], dtype=torch.float32).squeeze()
        x0 = torch.randn_like(self.x1)
        
        # T = torch.tensor(self.T * torch.ones(self.x1.shape[0]), dtype=torch.float32)
        T = torch.tensor([self.T] * self.x1.shape[0])

        return self.process(x0, T)

    def process(self, x0: torch.Tensor, T: torch.tensor) -> torch_geometric.data.Batch:
        """## Process the data.

        ### Args:
            - `x (torch.Tensor)`: input data.

        ### Returns:
            - `torch_geometric.data.Batch`: processed data.
        """
        print(x0.mean())
        x0 = x0 - torch.mean(x0, axis=0)
        print(x0.mean())

        # if self.align:
        #     r = Rotation.align_vectors(a=x1.numpy(), b=x0.numpy(), weights=None)[0]  # create rotation of x0 to x1:s reference frame
        #     x0 = torch.tensor(r.apply(x0.numpy()), dtype=torch.float32)

        datalist = [torch_geometric.data.Data(x=torch.zeros_like(x0), x0=x0, x1=self.x1, T=T, atom_number=self.atom_order, bond_index=self.bond_index, bonds=self.bonds)]
        batch = torch_geometric.data.Batch.from_data_list(datalist)

        batch = self.add_radius_graph(batch)
        batch = self.add_bond_graph(batch)
        batch = self.coalesce(batch)
        return batch



def get_mdqm9_trajs(temp_index, traj_filename, traj_path: str, scale: bool, split: str) -> np.array:
    """## Load MDQM9 trajectories.

    ### Args:
        - `temp_index (_type_)`: index linked to temperature.
        - `traj_filename (_type_)`: name of the trajectory file.
        - `traj_path (str)`: path to the trajectory file.
        - `scale (bool)`: if True, the data is scaled before processing.
        - `split (str)`: which data split to load (e.g. train/test/val).

    ### Returns:
        - `np.array`: the trajectory data at given temperature.
    """

    trajs = np.load(os.path.join(traj_path, split, traj_filename))[temp_index, :, :, :]

    # calculate center of mass and center the data
    coms = np.mean(trajs, axis=1)
    trajs = trajs - coms[:, np.newaxis, :]

    if scale:
        trajs = trajs * SCALING_FACTOR
        trajs = np.array(trajs)
    return trajs


def get_mdqm9_atom_numbers(traj_path, sdf_path, traj_filename, sdf_filename, distinguish: bool, split: str) -> torch.tensor:
    """## _summary_

    ### Args:
        - `traj_path (_type_)`: path to the trajectory file.
        - `sdf_path (_type_)`: path to the sdf file.
        - `traj_filename (_type_)`: name of the trajectory file.
        - `sdf_filename (_type_)`: name of the sdf file.
        - `distinguish (bool)`: whether to use distinguishable atoms (e.g. 0, 1, 2, 3, ...) or actual atom numbers.
        - `split (str)`: which data split to load (e.g. train/test/val).

    ### Returns:
        - `torch.tensor`: _description_
    """

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
    """## Obtain indexes of bond connections and bond types in the molecule.

    ### Args:
        - `traj_filename (str)`: name of the trajectory file.
        - `sdf_path (str)`: path to the sdf file.
        - `sdf_filename (str)`: name of the sdf file.

    ### Returns:
        - `tuple`: bond index and bonds.
    """

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



if __name__ == "__main__":
    config = utils.load_config("paper/bg_thermo", "settings.json")
    # dataset = MDQM9MultiTempDataset(traj_filename=config.mdqm9_traj_filename, 
    #                        sdf_filename="mdqm9.sdf", 
    #                        traj_path="molecular_interpolants/replica_exchange_trajs/rotated_replica_exchange_trajs/", 
    #                        sdf_path="molecular_interpolants/replica_exchange_trajs/", 
    #                        split='train', 
    #                        Ts=config.T, 
    #                        scale=config.scale_trajs, 
    #                        cutoff=config.cutoff)
    # print(len(dataset)) 
    # dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    # print(next(iter(dataloader)))
    # batch = next(iter(dataloader))
    # print(batch.x0)
    # print(batch.T)
    dataset = SamplerDataset(
        traj_filename=config.mdqm9_traj_filename, 
        sdf_filename="mdqm9.sdf", 
        traj_path="molecular_interpolants/replica_exchange_trajs/rotated_replica_exchange_trajs/", 
        sdf_path="molecular_interpolants/replica_exchange_trajs/", 
        split='test', 
        Ts=config.T_sampler, 
        n_samples=config.n_samples,
        scale=config.scale_trajs, 
        cutoff=config.cutoff,
        align=config.align,
    )
    
    test_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    print(next(iter(test_loader)))
