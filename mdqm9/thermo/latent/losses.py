import torch
import torch_geometric

from typing import Callable
from thermo.latent.interpolants import BaseInterpolant


class BaseVelocityLoss(torch.nn.Module):
    """## Base Class for Velocity Losses
    """

    def __init__(self, interpolant: BaseInterpolant, t_distr="uniform") -> None:
        """## Initialize the loss function.

        ### Args:
            - `interpolant (BaseInterpolant)`: interpolant to use to compute the loss
            - `t_distr (str)`: distribution of time points to use for interpolation. Default is "uniform".

        ### Raises:
            - `AssertionError`: if the time distribution is not "uniform" or "beta"
        """

        super(BaseVelocityLoss, self).__init__()

        assert t_distr in ["uniform", "beta"], f"Invalid time distribution: {t_distr}"
        
        self.interpolant = interpolant
        self.t_distr = t_distr

    def forward(self, batch: torch_geometric.data.Batch, b: torch.nn.Module) -> torch.Tensor:
        """## Compute the loss.

        ### Args:
            - `batch0 (torch_geometric.data.Batch)`: batch with data at t=0.
            - `batch1 (torch_geometric.data.Batch)`: batch with data at t=1.
            - `b (torch.nn.Module)`: velocity field neural network to use to compute the loss.

        ### Returns:
            - `torch.Tensor`: loss value.
        """

        x0 = batch.x0
        x1 = batch.x1

        if self.t_distr == "uniform":
            t = torch.cat([torch.rand(1).repeat(len(data.atom_number)) for data in batch.to_data_list()]).unsqueeze(1).to(x0.device)
        elif self.t_distr == "beta":
            distr = torch.distributions.beta.Beta(2, 1)

            t = torch.cat([distr.sample((1,)).repeat(len(data.atom_number)) for data in batch.to_data_list()]).unsqueeze(1).to(x0.device)
        else:
            raise ValueError(f"Invalid value of time distribution: {self.t_distr}")
        
        xtp, xtm, z = self.interpolant.calc_antithetic_xts(t, x0, x1)

        xtp = xtp - torch.mean(xtp, dim=0)
        xtm = xtm - torch.mean(xtm, dim=0)

        batch_plus = batch.clone()
        batch_minus = batch.clone()

        batch_plus.x = xtp
        batch_minus.x = xtm

        batch_plus.x0 = x0
        batch_minus.x0 = x0

        batch_plus.t = t.squeeze()
        batch_minus.t = t.squeeze()

        batch_plus = b(batch_plus)
        batch_minus = b(batch_minus)

        bt_plus = batch_plus.output
        bt_minus = batch_minus.output

        # calc loss
        loss_fn = self.make_batch_loss()
        loss_val = loss_fn(t, z, x0, x1, bt_plus, bt_minus).mean()
        return loss_val
    
    def loss_per_sample(self, t: torch.tensor, z: torch.tensor, x0: torch.tensor, x1: torch.tensor, bt_plus: torch.tensor, bt_minus: torch.tensor) -> torch.Tensor:
        """## Compute the loss for a single sample.

        ### Args:
            - `t (torch.tensor)`: time at which to evaluate the interpolant.
            - `z (torch.tensor)`: gaussian noise used in interpolant.
            - `x0 (torch.tensor)`: initial coordinates (at t=0).
            - `x1 (torch.tensor)`: final coordinates (at t=1).
            - `bt_plus (torch.tensor)`: velocity field evaluated at the interpolated coordinates in the positive direction.
            - `bt_minus (torch.tensor)`: velocity field evaluated at the interpolated coordinates in the negative direction.

        ### Raises:
            - `NotImplementedError`: by default, this method should be overridden by subclasses.

        ### Returns:
            - `torch.tensor`: loss value per sample.
        """

        raise NotImplementedError

    def make_batch_loss(self) -> Callable:
        """## Vectorize the loss function.

        ### Returns:
            - `Callable`: vectorized loss function.
        """

        in_dims = (0, 0, 0, 0, 0, 0)  # share batch dimension for t, z, x0, x1, bt_plus, bt_minus
        batched_loss = torch.vmap(self.loss_per_sample, in_dims=in_dims, randomness='different')
        return batched_loss


class OneSidedVelocityLoss(BaseVelocityLoss):
    def __init__(self, interpolant: BaseInterpolant, t_distr="uniform") -> None:
        super(OneSidedVelocityLoss, self).__init__(interpolant, t_distr)

    def loss_per_sample(self, t, z, x0, x1, bt_plus, bt_minus):
        dtIt = self.interpolant.dtIt(t, x0, x1)
        
        loss = 0.5*torch.sum(bt_plus**2) - torch.sum((dtIt) * bt_plus)
        return loss