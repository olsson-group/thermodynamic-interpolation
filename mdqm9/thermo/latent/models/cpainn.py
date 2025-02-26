import warnings

import torch
import torch_geometric
from torch_scatter import scatter

from thermo.latent.models import device, embedding, graph


class cPaiNN(device.Module):
    """## SE(3)-equivariant ChiroPaiNN model. 
    """

    @property
    def device(self) -> torch.device:
        """## Get the device of the model.

        ### Returns:
            - `torch.device`: device of the model.
        """
        return next(self.parameters()).device

    def __init__(
        self,
        n_features: int=32,
        score_layers: int=5,
        n_types=25,
        time_length=10,
        temp_length=10,
        temperatures=[300, 400, 500, 600, 700, 800, 900, 1000],
    ):
        """## Initialize the model.

        ### Args:
            - `n_features (int, optional)`: number of features. Defaults to 32.
            - `score_layers (int, optional)`: number of score layers. Defaults to 5.
            - `n_types (int, optional)`: number of "atom" types. Defaults to 25.
            - `time_length (int, optional)`: length of time embedding. Defaults to 10.
        """

        super().__init__()

        if len(temperatures) > 1:
            layers = [
                graph.AddSpatialFeatures(),
                graph.AddEquivariantFeatures(n_features),
                embedding.NominalEmbedding("edge_type", n_features, n_types=4, feature_type="edge"),
                embedding.NominalEmbedding("atom_number", n_features, n_types=n_types),
                embedding.TemperatureEmbedding(feature_name="T", 
                                           temperatures=temperatures, 
                                           n_features=n_features, 
                                           max_length=temp_length),
                embedding.PositionalEmbedding("t", n_features, length=time_length),
                embedding.CombineInvariantFeatures(3*n_features, n_features),
                PaiNNBase(
                    n_features=n_features,
                    n_layers=score_layers,
                ),
            ]
        else:
            layers = [
                graph.AddSpatialFeatures(),
                graph.AddEquivariantFeatures(n_features),
                embedding.NominalEmbedding("edge_type", n_features, n_types=4, feature_type="edge"),
                embedding.NominalEmbedding("atom_number", n_features, n_types=n_types),
                embedding.PositionalEmbedding("t", n_features, length=time_length),
                embedding.CombineInvariantFeatures(2*n_features, n_features),
                PaiNNBase(
                    n_features=n_features,
                    n_layers=score_layers,
                ),
            ]
        

        """self.net = torch.nn.Sequential(
            graph.AddSpatialFeatures(),
            graph.AddEquivariantFeatures(n_features),
            embedding.NominalEmbedding("edge_type", n_features, n_types=4, feature_type="edge"),
            embedding.NominalEmbedding("atom_number", n_features, n_types=n_types),
            embedding.TemperatureEmbedding(feature_name="T", 
                                           temperatures=temperatures, 
                                           n_features=n_features, 
                                           max_length=temp_length),
            embedding.PositionalEmbedding("t", n_features, length=time_length),
            embedding.CombineInvariantFeatures(3*n_features, n_features),
            PaiNNBase(
                n_features=n_features,
                n_layers=score_layers,
            ),
        )"""

        self.net = torch.nn.Sequential(*layers)


    def forward(self, batch: torch_geometric.data.Batch) -> torch_geometric.data.Batch:
        """## Forward pass of the model.

        ### Args:
            - `batch (torch_geometric.data.Batch)`: input batch.

        ### Returns:
            - `torch_geometric.data.Batch`: output batch.
        """
        
        updated_batch = self.net(batch)
        batch.output = updated_batch.equivariant_node_features.squeeze()
        
        return batch


class PaiNNBase(device.Module):
    """## PaiNN base model.
    """

    def __init__(
        self,
        n_features=128,
        n_layers=5,
        n_features_out=1,
        length_scale=10,
    ):
        """## Initialize the PaiNN base model.

        ### Args:
            - `n_features (int, optional)`: number of features. Defaults to 128.
            - `n_layers (int, optional)`: number of layers. Defaults to 5.
            - `n_features_out (int, optional)`: number of output features. Defaults to 1.
            - `length_scale (int, optional)`: length scale. Defaults to 10.
        """

        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(
                SE3Message(
                    n_features=n_features,
                    length_scale=length_scale,
                )
            )
            layers.append(Update(n_features))

        layers.append(LayerReadout(n_features, n_features_out))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, batch: torch_geometric.data.Batch) -> torch_geometric.data.Batch:
        """## Forward pass of the PaiNN base model.

        ### Args:
            - `batch (torch_geometric.data.Batch)`: input batch.

        ### Returns:
            - `torch_geometric.data.Batch`: output batch.
        """
        return self.layers(batch)


class Message(device.Module):
    """## Message module.
    """

    def __init__(
        self,
        n_features=128,
        length_scale=10,
        use_edge_features=True,
    ):
        """## Initialize the message module.

        ### Args:
            - `n_features (int, optional)`: number of features. Defaults to 128.
            - `length_scale (int, optional)`: length scale. Defaults to 10.
            - `use_edge_features (bool, optional)`: whether to use edge features. Defaults to True.
        """

        super().__init__()
        self.n_features = n_features
        self.use_edge_features = use_edge_features

        self.positional_encoder = embedding.PositionalEncoder(
            n_features, max_length=length_scale
        )

        phi_in_features = 2 * n_features if use_edge_features else n_features
        self.phi = embedding.MLP(phi_in_features, n_features, 4 * n_features)
        self.w = embedding.MLP(n_features, n_features, 4 * n_features)

    def forward(self, batch: torch_geometric.data.Batch) -> torch_geometric.data.Batch:
        """## Forward pass of the message module.

        ### Args:
            - `batch (torch_geometric.data.Batch)`: input batch.

        ### Returns:
            - `torch_geometric.data.Batch`: output batch.
        """
        src_node = batch.edge_index[0]
        dst_node = batch.edge_index[1]

        in_features = batch.invariant_node_features[src_node]

        if self.use_edge_features:
            in_features = torch.cat(
                [in_features, batch.invariant_edge_features], dim=-1
            )

        positional_encoding = self.positional_encoder(batch.edge_dist)

        gates, scale_edge_dir, ds, de = torch.split(
            self.phi(in_features) * self.w(positional_encoding),
            self.n_features,
            dim=-1,
        )
        gated_features = multiply_first_dim(
            gates, batch.equivariant_node_features[src_node]
        )
        scaled_edge_dir = multiply_first_dim(
            scale_edge_dir, batch.edge_dir.unsqueeze(1).repeat(1, self.n_features, 1)
        )

        dv = scaled_edge_dir + gated_features
        dv = scatter(dv, dst_node, dim=0)
        ds = scatter(ds, dst_node, dim=0)

        batch.equivariant_node_features += dv
        batch.invariant_node_features += ds
        batch.invariant_edge_features += de

        return batch


class SE3Message(device.Module):
    """## SE(3)-equivariant message module.
    """

    def __init__(
        self,
        n_features=128,
        length_scale=10,
    ):
        """## Initialize the SE(3)-equivariant message module.

        ### Args:
            - `n_features (int, optional)`: number of features. Defaults to 128.
            - `length_scale (int, optional)`: length scale. Defaults to 10.
        """

        super().__init__()
        self.n_features = n_features

        self.positional_encoder = embedding.PositionalEncoder(n_features, max_length=length_scale)

        phi_in_features = 2 * n_features
        self.phi = embedding.MLP(phi_in_features, n_features, 5 * n_features)
        self.w = embedding.MLP(n_features, n_features, 5 * n_features)

    def forward(self, batch: torch_geometric.data.Batch) -> torch_geometric.data.Batch:
        """## Forward pass.

        ### Args:
            - `batch (torch_geometric.data.Batch)`: input batch.

        ### Returns:
            - `torch_geometric.data.Batch`: output batch.
        """

        src_node = batch.edge_index[0]
        dst_node = batch.edge_index[1]

        in_features = torch.cat(
            [
                batch.invariant_node_features[src_node],
                batch.invariant_edge_features,
            ],
            dim=-1,
        )

        positional_encoding = self.positional_encoder(batch.edge_dist)

        gates, scale_edge_dir, ds, de, cross_product_gates = torch.split(
            self.phi(in_features) * self.w(positional_encoding),
            self.n_features,
            dim=-1,
        )
        gated_features = multiply_first_dim(gates, batch.equivariant_node_features[src_node])
        scaled_edge_dir = multiply_first_dim(
            scale_edge_dir, batch.edge_dir.unsqueeze(1).repeat(1, self.n_features, 1)
        )

        dst_node_edges = batch.edge_dir.unsqueeze(1).repeat(1, self.n_features, 1)
        src_equivariant_node_features = batch.equivariant_node_features[dst_node]
        cross_produts = torch.cross(dst_node_edges, src_equivariant_node_features, dim=-1)

        gated_cross_products = multiply_first_dim(cross_product_gates, cross_produts)

        dv = scaled_edge_dir + gated_features + gated_cross_products
        dv = scatter(dv, dst_node, dim=0)
        ds = scatter(ds, dst_node, dim=0)

        batch.equivariant_node_features = batch.equivariant_node_features + dv
        batch.invariant_node_features = batch.invariant_node_features + ds
        batch.invariant_edge_features = batch.invariant_edge_features + de

        return batch


def multiply_first_dim(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """## Multiply the first dimension of two tensors.

    ### Args:
        - `w (torch.Tensor)`: first tensor.
        - `x (torch.Tensor)`: second tensor.

    ### Returns:
        - `torch.Tensor`: result of the multiplication.
    """

    with warnings.catch_warnings(record=True):
        return (w.T * x.T).T


class Update(device.Module):
    """## Update module.
    """

    def __init__(self, n_features=128) -> None:
        """## Initialize the update module.

        ### Args:
            - `n_features (int, optional)`: number of features. Defaults to 128.
        """

        super().__init__()
        self.u = EquivariantLinear(n_features, n_features)
        self.v = EquivariantLinear(n_features, n_features)
        self.n_features = n_features
        self.mlp = embedding.MLP(2 * n_features, n_features, 3 * n_features)

    def forward(self, batch: torch_geometric.data.Batch) -> torch_geometric.data.Batch:
        """## Forward pass of the update module.

        ### Args:
            - `batch (torch_geometric.data.Batch)`: input batch.

        ### Returns:
            - `torch_geometric.data.Batch`: output batch.
        """

        v = batch.equivariant_node_features
        s = batch.invariant_node_features

        vv = self.v(v)
        uv = self.u(v)

        vv_norm = vv.norm(dim=-1)
        vv_squared_norm = vv_norm**2

        mlp_in = torch.cat([vv_norm, s], dim=-1)

        gates, scale_squared_norm, add_invariant_features = torch.split(
            self.mlp(mlp_in), self.n_features, dim=-1
        )

        delta_v = multiply_first_dim(uv, gates)
        delta_s = vv_squared_norm * scale_squared_norm + add_invariant_features

        batch.invariant_node_features = batch.invariant_node_features + delta_s
        batch.equivariant_node_features = batch.equivariant_node_features + delta_v

        return batch


class EquivariantLinear(device.Module):
    """## Equivariant linear layer.
    """

    def __init__(self, n_features_in: int, n_features_out: int) -> None:
        """## Initialize an equivariant linear layer.

        ### Args:
            - `n_features_in (int)`: number of input features.
            - `n_features_out (int)`: number of output features.
        """

        super().__init__()
        self.linear = torch.nn.Linear(n_features_in, n_features_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """## Forward pass.

        ### Args:
            - `x (torch.Tensor)`: input tensor.

        ### Returns:
            - `torch.Tensor`: output tensor.
        """
        return self.linear(x.swapaxes(-1, -2)).swapaxes(-1, -2)


class LayerReadout(device.Module):
    """## Layer readout module.
    """

    def __init__(self, n_features=128, n_features_out=13) -> None:
        """## Initialize the layer readout module.

        ### Args:
            - `n_features (int, optional)`: number of input features. Defaults to 128.
            - `n_features_out (int, optional)`: number of output features. Defaults to 13.
        """
        
        super().__init__()
        self.mlp = embedding.MLP(n_features, n_features, 2 * n_features_out)
        self.V = EquivariantLinear(  # pylint:disable=invalid-name
            n_features, n_features_out
        )
        self.n_features_out = n_features_out

    def forward(self, batch):
        invariant_node_features_out, gates = torch.split(
            self.mlp(batch.invariant_node_features), self.n_features_out, dim=-1
        )

        equivariant_node_features = self.V(batch.equivariant_node_features)
        equivariant_node_features_out = multiply_first_dim(
            equivariant_node_features, gates
        )

        batch.invariant_node_features = invariant_node_features_out
        batch.equivariant_node_features = equivariant_node_features_out
        return batch


'''class FinalReadout(torch.nn.Module):
    """
    Final readout module. Merges equivariant and invariant features.
    """
    def __init__(self, n_features: int=32) -> None:
        """
        Initialize the final readout module.
        :param n_features: The number of input features.
        :return: None
        """
        super().__init__()
        # self.mlp = embedding.MLP(n_features, n_features, n_features)
        # self.V = EquivariantLinear(n_features, n_features)

    def forward(self, batch):
        """
        Forward pass of the final readout module.
        :param batch: The input batch.
        :return: The batch with updated combined equivariant and invariant features.
        """

        # invariant_node_features_out = self.mlp(batch.invariant_node_features)
        # equivariant_node_features_out = self.V(batch.equivariant_node_features)

        batch.equivariant_node_features = batch.equivariant_node_features.squeeze()
        batch.output = batch.equivariant_node_features + multiply_first_dim(batch.equivariant_node_features, batch.invariant_node_features)
        return batch'''



