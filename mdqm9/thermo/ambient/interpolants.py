import torch
import torch.nn as nn


class BaseInterpolant(nn.Module):
    """## Base class for interpolants.
    """

    def __init__(self):
        """## Initialize the interpolant.
        """
        super(BaseInterpolant, self).__init__()

        self.gamma = None
        self.It = None

    def calc_antithetic_xts(self, t: torch.tensor, x0: torch.tensor, x1: torch.tensor) -> tuple:
        """## Compute interpolated 3D cartesian coordinates of molecule at time t. Antithetic sampling is used to reduce variance of the interpolant.

        ### Args:
            - `t (torch.tensor)`: time
            - `x0 (torch.tensor)`: initial atom 3D-positions
            - `x1 (torch.tensor)`: final atom 3D-positions

        ### Returns:
            - `tuple`: interpolated 3D-positions at time t (positive and negative directions), gaussian noise of the interpolant
        """

        z = torch.randn(x0.shape).to(t)
        gamma = self.gamma(t)
        It = self.It(t, x0, x1)
        return It + gamma*z, It - gamma*z, z
    
    def calc_regular_xt(self, t: torch.tensor, x0: torch.tensor, x1: torch.tensor) -> tuple:
        """## Compute interpolated 3D cartesian coordinates of molecule at time t.

        ### Args:
            - `t (torch.tensor)`: time
            - `x0 (torch.tensor)`: initial atom 3D-positions
            - `x1 (torch.tensor)`: final atom 3D-positions

        ### Returns:
            - `tuple`: interpolated 3D-positions at time t, gaussian noise of the interpolant
        """

        z = torch.randn(x0.shape).to(t)
        return self.It(t, x0, x1) + self.gamma(t)*z, z

    def forward(self):
        raise NotImplementedError


class LinearInterpolant(BaseInterpolant):
    """## Linear interpolant.
    """

    def __init__(self, a: float=1, gamma: str='brownian') -> None:
        """## Initialize the linear interpolant.

        ### Args:
            - `a (float, optional)`: variance-related parameter for brownian noise. Defaults to 1.
            - `gamma (str, optional)`: type of gamma function. Defaults to 'brownian'.

        ### Raises:
            - `NotImplementedError`: if the requested gamma function is not implemented.
        """

        super(LinearInterpolant, self).__init__()

        # brownian gamma
        if gamma == 'brownian':
            a = torch.tensor(a)
            
            self.gamma = lambda t: torch.sqrt(a*t*(1-t))
            self.gamma_dot = lambda t: (1/(2*torch.sqrt(a*t*(1-t)))) * a*(1 -2*t)
            self.gg_dot = lambda t: (a/2)*(1-2*t)

        # sine-squared gamma
        elif gamma == 'sin2':
            self.gamma = lambda t: torch.sin(torch.pi * t)**2
            self.gamma_dot = lambda t: 2*torch.pi*torch.sin(torch.pi * t)*torch.cos(torch.pi*t)
            self.gg_dot = lambda t: self.gamma(t)*self.gamma_dot(t)

        elif gamma == 'sig_sum':
            a = torch.tensor(a)
            scale_factor = torch.tensor(2.2)
            self.gamma = lambda t: scale_factor*(torch.sigmoid(a*(t-(1/2)) + 1) - torch.sigmoid(a*(t-(1/2)) - 1) - torch.sigmoid((-a/2) + 1) + torch.sigmoid((-a/2) - 1))
            self.gamma_dot = lambda t: scale_factor*((-a)*( 1 - torch.sigmoid(-1 + a*(t - (1/2))) )*torch.sigmoid(-1 + a*(t - (1/2)))  + a*(1 - torch.sigmoid(1 + a*(t - (1/2)))  )*torch.sigmoid(1 + a*(t - (1/2))))
            self.gg_dot = lambda t: self.gamma(t)*self.gamma_dot(t)

        else:
            raise NotImplementedError 

        # linear interpolant
        self.a = lambda t: (1-t)
        self.adot = lambda t: -1.0

        self.b = lambda t: t
        self.bdot = lambda t: 1.0

        self.It = lambda t, x0, x1: self.a(t)*x0 + self.b(t)*x1
        self.dtIt = lambda t, x0, x1: self.adot(t)*x0 + self.bdot(t)*x1

    def calc_regular_xt(self, t: torch.tensor, x0: torch.tensor, x1: torch.tensor) -> tuple:
        return super().calc_regular_xt(t, x0, x1)

    def calc_antithetic_xts(self, t: torch.tensor, x0: torch.tensor, x1:torch.tensor) -> tuple:
        return super().calc_antithetic_xts(t, x0, x1)
