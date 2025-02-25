import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import argparse

import torch
import torch_geometric

from thermo.ambient.models import device


def write_json(data: dict, path: str, filename: str) -> None:
    """## Write data to a JSON file.

    ### Args:
        - `data (dict)`: data to write.
        - `path (str)`: path to save the JSON file.
        - `filename (str)`: name of the JSON file.
    """
    
    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, f"{filename}.json"), 'w') as f:
        json.dump(data, f, indent=4)
    f.close()


def load_config(path: str, filename: str) -> argparse.Namespace:
    """## Load configuration settings from a JSON file into argparser.

    ### Args:
        - `path (str)`: path to the JSON file.
        - `filename (str)`: name of the JSON file.

    ### Returns:
        - `argparse.Namespace`: configuration settings.
    """
    
    settings = json.load(open(os.path.join(path, filename), 'r'))
    parser = argparse.ArgumentParser()
    
    for key, value in settings.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)
    return parser.parse_args()


def clone_config(save_path: str, filename: str, config: argparse.Namespace) -> None:
    """## Clone configuration settings into a JSON file.

    ### Args:
        - `save_path (str)`: path to save the JSON file.
        - `filename (str)`: name of the JSON file.
        - `config (argparse.Namespace)`: configuration settings.
    """
    
    if not os.path.exists(os.path.join(save_path, filename)):
        os.makedirs(os.path.join(save_path, filename))

    with open(os.path.join(save_path, filename, 'settings.json'), 'w') as f:
        json.dump(vars(config), f, indent=4)
    f.close()


# TODO: fix stuff below

class Coalesce(device.DeviceTracker):
    # TODO: make into one function along with AddEdges, AddBondGraph, AddRadiusGraph
    def forward(self, batch):
        batch = batch.clone()
        assert hasattr(batch, "edge_type"), "Edge types have not been defined on batch"
        batch.edge_index, batch.edge_type = torch_geometric.utils.coalesce(
            batch.edge_index,
            batch.edge_type,
            reduce="max",
        )

        return batch


class AddEdges(device.DeviceTracker):
    # TODO: make into one function along with AddEdges, AddBondGraph, AddRadiusGraph
    def __init__(self):
        super().__init__()

    def get_edges(self, batch):
        raise NotImplementedError()

    def forward(self, batch):
        batch = batch.clone()
        device = batch.x.device
        edge_index, edge_type = self.get_edges(batch)

        if batch.edge_index is not None:
            edge_index = torch.cat([batch.edge_index, edge_index], dim=1)
            edge_type = torch.cat([batch.edge_type, edge_type], dim=0)

        batch.edge_index = edge_index.to(device)
        batch.edge_type = edge_type.to(device)

        return batch


class AddBondGraph(AddEdges):
    # TODO: make into one function along with AddEdges, AddBondGraph, AddRadiusGraph
    def get_edges(self, batch):
        return batch.bond_index, batch.bonds


class AddRadiusGraph(AddEdges):
    # TODO: make into one function along with AddEdges, AddBondGraph, AddRadiusGraph
    def __init__(self, cutoff=None, edge_type=0):
        super().__init__()
        self.cutoff = float("inf") if cutoff is None else cutoff
        self.edge_type = edge_type

    def get_edges(self, batch):
        device = batch.x.device

        edge_index = torch_geometric.nn.radius_graph(batch.x, r=self.cutoff, batch=batch.batch, max_num_neighbors=999999)
        edge_type = torch.ones(edge_index.shape[1], dtype=torch.long, device=device) * self.edge_type

        return edge_index, edge_type

