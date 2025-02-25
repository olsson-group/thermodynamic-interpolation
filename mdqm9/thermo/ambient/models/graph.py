import torch
import torch_geometric

from thermo.ambient.models import device


# TODO: remove

class AddSpatialFeatures(device.Module):
    """## Add distances and their directions to batch.
    """
    def forward(self, batch: torch_geometric.data.Batch) -> torch_geometric.data.Batch:
        """## Forward pass.

        ### Args:
            - `batch (torch_geometric.data.Batch)`: input batch.

        ### Asserts:
            - `hasattr(batch, "edge_type"), "Edge types have not been defined on batch"`

        ### Returns:
            - `torch_geometric.data.Batch`: output batch.
        """
        
        batch = batch.clone()
        assert hasattr(batch, "edge_type"), "Edge types have not been defined on batch"
        r = batch.x[batch.edge_index[0]] - batch.x[batch.edge_index[1]]
        edge_dist = r.norm(dim=-1)
        edge_dir = r / (1 + edge_dist.unsqueeze(-1))

        batch.edge_dist = edge_dist
        batch.edge_dir = edge_dir
        return batch


class AddEquivariantFeatures(device.DeviceTracker):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features

    def forward(self, batch):
        eq_features = torch.zeros(
            batch.atoms.shape[0],
            self.n_features,
            3,
        )
        batch.equivariant_node_features = eq_features.to(self.device)
        return batch