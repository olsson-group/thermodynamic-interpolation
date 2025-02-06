import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn


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

    def forward(self, integration_time: torch.tensor, states: tuple, x0s, beta0s: torch.tensor, beta1s: torch.tensor) -> tuple:

        """
        Forward pass of the ODE wrapper. This is the function that is called by the torchdiffeq odeint_adjoint.
        :param integration_time: The time at which the ODE is evaluated.
        :param states: Current integration states, i.e. (x_{t-1}, dlogp_{t-1}).
        :param batch: Torch Geometric batch of molecular data.
        :return: Tuple of value and divergence of the vector field.
        """

        if self.return_dlogp:
            xs, _ = states
            ts = torch.ones_like(xs)*integration_time

            b = self.b(x0s, xs, ts, beta0s, beta1s)
            divergence = self.compute_divergence(self.b, x0s, xs, ts, beta0s, beta1s)
            return (b, -divergence) if not self.reverse_ode else (-b, divergence)
        
        else:
            xs = states
            ts = torch.ones_like(xs)*integration_time
            b = self.b(x0s, xs, ts, beta0s, beta1s)
            return b

    @staticmethod
    def compute_divergence(b: nn.Module, x0s, xs: torch.tensor, ts: torch.tensor, beta0s: torch.tensor, beta1s: torch.tensor) -> torch.tensor:
        bs = xs.shape[0]

        with torch.set_grad_enabled(True):
            xs.requires_grad_(True)
            ts.requires_grad_(True)
            
            b_val = b(x0s, xs, ts, beta0s, beta1s)
            divergence = 0.0

            for i in range(xs.shape[1]):
                divergence += torch.autograd.grad(b_val[:, i].sum(), xs, create_graph=True)[0][:, i]
        return divergence.view(bs)*1e-2

