import torch

import thermo.utils as utils

from typing import Callable


class BaseVelocityLoss(torch.nn.Module):
    def __init__(self, interpolant) -> None:
        super(BaseVelocityLoss, self).__init__()
        
        self.interpolant = interpolant

    def forward(self, b, x0s, x1s, beta0s, beta1s):
        xtps, xtms, ts, beta0s, beta1s, zs = utils.get_interpolated_batch(x0s, x1s, beta0s, beta1s, self.interpolant)

        # get bt_plus and bt_minus
        btps = b(x0s, xtps, ts, beta0s, beta1s)
        btms = b(x0s, xtms, ts, beta0s, beta1s)

        # calc loss
        loss_fn = self.make_batch_loss()
        loss_val = loss_fn(ts, zs, x0s, x1s, btps, btms).mean()
        return loss_val
    
    def loss_per_sample(self, ts, zs, x0s, x1s, btps, btms):
        raise NotImplementedError

    def make_batch_loss(self) -> Callable:
        in_dims = (0, 0, 0, 0, 0, 0)  # share batch dimension for t, z, x0, x1, bt_plus, bt_minus
        batched_loss = torch.vmap(self.loss_per_sample, in_dims=in_dims, randomness='different')
        return batched_loss


class OneSidedVelocityLoss(BaseVelocityLoss):
    def __init__(self, interpolant) -> None:
        super(OneSidedVelocityLoss, self).__init__(interpolant)

    def loss_per_sample(self, t, z, x0, x1, bt_plus, bt_minus):
        dtIt = self.interpolant.dtIt(t, x0, x1)
        
        loss = 0.5*torch.sum(bt_plus**2) - torch.sum((dtIt) * bt_plus)
        return loss


class StandardVelocityLoss(BaseVelocityLoss):
    def __init__(self, interpolant) -> None:
        super(StandardVelocityLoss, self).__init__(interpolant)

    def loss_per_sample(self, ts, zs, x0s, x1s, btps, btms):
        dtIt = self.interpolant.dtIt(ts, x0s, x1s)
        gamma_dot = self.interpolant.gamma_dot(ts)

        # compute loss from two "directions" for improved numerical stability
        loss = 0.5*torch.sum(btps**2) - torch.sum((dtIt + gamma_dot*zs) * btps)
        loss += 0.5*torch.sum(btms**2) - torch.sum((dtIt - gamma_dot*zs) * btms)
        return loss
