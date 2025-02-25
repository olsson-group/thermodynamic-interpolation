import torch
import torch_geometric as geom

from thermo.latent.models import device


class AddEdges(device.DeviceTracker):
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


class AddRadiusGraph(AddEdges):
    def __init__(self, cutoff=None, edge_type=0):
        super().__init__()
        self.cutoff = float("inf") if cutoff is None else cutoff
        self.edge_type = edge_type

    def get_edges(self, batch):
        device = batch.x.device

        edge_index = geom.nn.radius_graph(batch.x, r=self.cutoff, batch=batch.batch, max_num_neighbors=999999)
        edge_type = torch.ones(edge_index.shape[1], dtype=torch.long, device=device) * self.edge_type

        return edge_index, edge_type


class AddBondGraph(AddEdges):
    def get_edges(self, batch):
        return batch.bond_index, batch.bonds


class ConnectVirtual(AddEdges):
    def __init__(self, edge_type=4):
        super().__init__()
        self.edge_type = edge_type

    def get_edges(self, batch):
        device = batch.x.device
        edge_index = geom.nn.radius_graph(batch.x, r=float("inf"), batch=batch.batch)

        virtual_edge_mask = torch.logical_or(
            batch.atoms[edge_index[0]] == 0,
            batch.atoms[edge_index[1]] == 0,
        )
        edge_index = edge_index[:, virtual_edge_mask]
        edge_type = torch.ones(edge_index.shape[1], dtype=torch.long, device=device) * self.edge_type

        return edge_index, edge_type


class Coalesce(device.DeviceTracker):
    def forward(self, batch):
        batch = batch.clone()
        assert hasattr(batch, "edge_type"), "Edge types have not been defined on batch"
        batch.edge_index, batch.edge_type = geom.utils.coalesce(
            batch.edge_index,
            batch.edge_type,
            reduce="max",
        )
        return batch


class AddVirtualNode(device.DeviceTracker):
    def forward(self, data):
        data = data.clone()
        device = data.x.device
        virtual_nodes_x = torch.zeros(1, 3, device=device)
        virtual_nodes_atoms = torch.zeros(1, dtype=torch.long, device=device)
        data.x = torch.cat([data.x, virtual_nodes_x], dim=0)
        data.atoms = torch.cat([data.atoms, virtual_nodes_atoms], dim=0)
        return data


class AddSpatialFeatures(device.Module):
    def forward(self, batch):
        batch = batch.clone()
        assert hasattr(batch, "edge_type"), "Edge types have not been defined on batch"

        r = batch.x[batch.edge_index[0]] - batch.x[batch.edge_index[1]]
        edge_dist = r.norm(dim=-1)
        edge_dir = r / (1 + edge_dist.unsqueeze(-1))

        batch.edge_dist = edge_dist
        batch.edge_dir = edge_dir
        return batch

class AddEquivariantFeatures(device.Module):
    """
    Module for adding equivariant features.
    """
    def __init__(self, n_features):
        """
        Initializes the class.
        :param n_features: The number of features.
        """
        super().__init__()
        self.n_features = n_features

    def forward(self, batch):
        """
        Forward pass of the module.
        :param batch: The input batch.
        :return: The batch with added equivariant features.
        """
        device = batch.x.device
        batch.equivariant_node_features = torch.zeros(batch.batch.shape[0], self.n_features, 3, 
                                                      dtype=torch.float32, device=device)
        return batch

class AddGraph(device.Module):
    def __init__(self, cutoff=None, virtual_node=False):
        super().__init__()
        self.create_graph = torch.nn.Sequential(
            *[
                AddVirtualNode() if virtual_node else torch.nn.Identity(),
                AddBondGraph(),
                AddRadiusGraph(cutoff=cutoff),
                ConnectVirtual() if virtual_node else torch.nn.Identity(),
                Coalesce(),
                AddSpatialFeatures(),
            ]
        )

    def forward(self, batch):
        return self.create_graph(batch)
