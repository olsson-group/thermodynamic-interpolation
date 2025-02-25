import torch
import torch.nn as nn
import torch_geometric


class ODEWrapper(nn.Module):
    """## Wrap sample and dlogp differential equations into a single PyTorch module that is compatible with the
    torchdiffeq format.
    """

    def __init__(self, b: nn.Module, return_dlogp=False, reverse_ode=False) -> None:
        """## Initialize the ODE wrapper.

        ### Args:
            - `b (nn.Module)`: neural network velocity field.
            - `return_dlogp (bool, optional)`: whether to return the dlogp or not (default: False).
            - `reverse_ode (bool, optional)`: whether to integrate reversed ODE or not (default: False).
        """

        super(ODEWrapper, self).__init__()

        self.b = b
        self.return_dlogp = return_dlogp
        self.reverse_ode = reverse_ode

    def forward(self, integration_time: torch.tensor, states: tuple, batch: torch_geometric.data.Batch, n_steps: list) -> tuple:
        """## Forward pass of the ODE wrapper. This is the function that is called by the torchdiffeq odeint_adjoint.

        ### Args:
            - `integration_time (torch.tensor)`: current integration time.
            - `states (tuple)`: tuple containing the current state of the system. If return_dlogp is True, it contains the current conformation and dlogp, else it contains only the current conformation.
            - `batch (torch_geometric.data.Batch)`: batch of molecular data.
            - `n_steps (list)`: number of integration steps.

        ### Returns:
            - `tuple`: output of the velocity field network and the NEGATIVE divergence of the vector field if return_dlogp is True, else only the output of the velocity field network is returned.
        """

        if self.return_dlogp:
            x, _ = states
        
            sample_batch = self.reset_batch(batch.clone(), x, integration_time)
            dlogp_batch = self.reset_batch(batch.clone(), x, integration_time)
            
            n_steps.append(n_steps[-1] + 1)

            b = self.b(sample_batch).output
            divergence = self.compute_divergence(self.b, dlogp_batch)
            return (b, -divergence) if not self.reverse_ode else (-b, divergence, n_steps)
        
        else:
            x = states
            n_steps.append(n_steps[-1] + 1)

            sample_batch = self.reset_batch(batch.clone(), x, integration_time)
            b = self.b(sample_batch).output
            return b

    @staticmethod
    def compute_divergence(b: nn.Module, batch: torch_geometric.data.Batch) -> torch.tensor:
        """## Compute the divergence of the vector field. See: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13.

        ### Args:
            - `b (nn.Module)`: neural network velocity field.
            - `batch (torch_geometric.data.Batch)`: batch of molecular data.

        ### Returns:
            - `torch.tensor`: divergence of the vector field.
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
            return divergence.contiguous()*1e-2

    @staticmethod
    def reset_batch(batch: torch_geometric.data.Batch, x: torch.tensor, integration_time: torch.tensor) -> torch_geometric.data.Batch:
        """## Reset the batch with new x and t.

        ### Args:
            - `batch (torch_geometric.data.Batch)`: batch of molecular data.
            - `x (torch.tensor)`: new x-value.
            - `integration_time (torch.tensor)`: new integration time.

        ### Returns:
            - `torch_geometric.data.Batch`: batch of molecular data with new x and t.
        """
        
        # remove invariant and equivariant features from previous batch (bad solution but temporary fix...)
        del batch.invariant_node_features
        del batch.equvariant_node_features

        # reset the batch with new x and t
        batch.x = x.clone()
        batch.t = integration_time*torch.ones_like(batch.atoms)
        return batch
    


