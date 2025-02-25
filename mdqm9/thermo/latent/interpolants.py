import torch
import torch.nn as nn


class BaseInterpolant(nn.Module):
    """
    Base class for interpolants.
    """
    def __init__(self):
        super(BaseInterpolant, self).__init__()

        self.gamma = None
        self.It = None

    def calc_antithetic_xts(self, t: torch.tensor, x0: torch.tensor, x1: torch.tensor):
        """
        Compute the interpolated states at time t for antithetic sampling.
        :param t: torch tensor, time
        :param x0: torch tensor, initial state
        :param x1: torch tensor, final state
        :return: tuple of torch tensors, interpolated states
        """
        z = torch.randn(x0.shape).to(t)
        gamma = self.gamma(t)
        It = self.It(t, x0, x1)
        return It + gamma*z, It - gamma*z, z
    
    def calc_regular_xt(self, t: torch.tensor, x0: torch.tensor, x1: torch.tensor):
        """
        Implement for each interpolant!
        :param t: torch tensor, time
        :param x0: torch tensor, initial state
        :param x1: torch tensor, final state
        """
        z = torch.randn(x0.shape).to(t)
        return self.It(t, x0, x1) + self.gamma(t)*z, z

    def forward(self, _):
        raise NotImplementedError


class OneSidedLinearInterpolant(BaseInterpolant):
    """
    One-sided linear interpolant.
    """
    def __init__(self) -> None:
        """
        Initialize the one-sided linear interpolant.
        :return: None
        """
        super(OneSidedLinearInterpolant, self).__init__()

        # one-sided linear interpolant
        self.a = lambda t: (1-t)
        self.adot = lambda t: -1.0
        self.b = lambda t: t
        self.bdot = lambda t: 1.0
        
        self.It = lambda t, x0, x1: self.a(t)*x0 + self.b(t)*x1
        self.dtIt = lambda t, x0, x1: self.adot(t)*x0 + self.bdot(t)*x1

    def calc_regular_xt(self, t: torch.tensor, x0: torch.tensor, x1: torch.tensor) -> torch.tensor:
        """
        Calculate the interpolated states at time t for one-sided linear interpolant.
        :param t: torch tensor, time
        :param x0: torch tensor, random standard Gaussian noise
        :param x1: torch tensor, final state
        :return: torch tensor, interpolated state
        """
        return self.It(t, x0, x1), x0
    
    def calc_antithetic_xts(self, t, x0, x1):
        It_p = self.b(t)*x1 + self.a(t)*x0
        It_m = self.b(t)*x1 - self.a(t)*x0
        return It_p, It_m, x0