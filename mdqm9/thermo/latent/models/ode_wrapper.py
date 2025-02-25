import torch
import torch.nn as nn
import torch_geometric


class ODEWrapper(nn.Module):
    
    """
    Wrap the stochastic interpolants sample ODE and dlogp ODE into a single PyTorch module that is compatible with the
    torchdiffeq format.
    """

    def __init__(self, b: nn.Module, return_dlogp=False, reverse_ode=False) -> None:

        """
        Initialize the ODE wrapper.
        :param b: The velocity field neural network.
        :param return_dlogp: If True, return the log sample prabilities as well.
        :param reverse_ode: If True, reverse the ODE direction.
        :return: None
        """

        super(ODEWrapper, self).__init__()

        self.b = b
        self.return_dlogp = return_dlogp
        self.reverse_ode = reverse_ode

    def forward(self, integration_time: torch.tensor, states: tuple, batch: torch_geometric.data.Batch) -> tuple:

        """
        Forward pass of the ODE wrapper. This is the function that is called by the torchdiffeq odeint_adjoint.
        :param integration_time: The time at which the ODE is evaluated.
        :param states: Current integration states, i.e. (x_{t-1}, dlogp_{t-1}).
        :param batch: Torch Geometric batch of molecular data.
        :return: Tuple of value and divergence of the vector field.
        """

        if self.return_dlogp:
            x, _ = states
        
            sample_batch = self.reset_batch(batch.clone(), x, integration_time)
            dlogp_batch = self.reset_batch(batch.clone(), x, integration_time)

            b = self.b(sample_batch).output
            divergence = self.compute_divergence(self.b, dlogp_batch)
            return (b, -divergence) if not self.reverse_ode else (-b, divergence)
        
        else:
            x = states

            sample_batch = self.reset_batch(batch.clone(), x, integration_time)
            b = self.b(sample_batch).output
            return b

    @staticmethod
    def compute_divergence(b: nn.Module, batch: torch_geometric.data.Batch) -> torch.tensor:

        """
        Compute the divergence of the vector field. See: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13.
        :param b: The velocity field network.
        :param batch: Torch Geometric batch of molecular data.
        :return: Divergence of the vector field.
        """

        with torch.set_grad_enabled(True):
            # batch.x0.requires_grad_(True)
            batch.x.requires_grad_(True)

            x = torch.stack([data.x for data in batch.to_data_list()]).unsqueeze(-1)
            batch_size, n_atoms, n_dim, _ = x.shape

            vector_field = b(batch).output
            vector_field = vector_field.view(batch_size, n_atoms, n_dim)  # TODO: check this

            divergence = torch.zeros(batch_size).to(vector_field.device)
            for i in range(n_atoms):
                for j in range(n_dim):

                    divergence += torch.autograd.grad(vector_field[:, i, j].sum(), batch.x, create_graph=True)[0].contiguous().view(batch_size, n_atoms, n_dim)[:, i, j].contiguous()

                    #div_tmp = torch.autograd.grad(vector_field[:, i, j].sum(), batch.x, create_graph=True)[0]
                    #div_tmp = div_tmp.view(batch_size, n_atoms, n_dim)

                    # divergence += sum_diag
            return divergence.contiguous()

    @staticmethod
    def reset_batch(batch, x, integration_time):
        # remove invariant and equivariant features from previous batch (bad solution but temporary fix...)
        del batch.invariant_node_features
        del batch.equvariant_node_features

        # reset the batch with new x and t
        batch.x = x.clone()
        batch.t = integration_time*torch.ones_like(batch.atom_number)
        return batch
    


