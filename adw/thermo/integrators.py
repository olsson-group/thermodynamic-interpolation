import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
from torchdiffeq import odeint_adjoint

from thermo.models.ode_wrapper import ODEWrapper


class StandardIntegrator:
    def __init__(self, b, method: str='dopri5', n_step: int=100, atol: float=1e-4, rtol: float=1e-4, 
                 start: float=0.0, end: float=1.0, return_dlogp=False) -> None:
        """
        Initialize the integrator.
        :param b: The velocity field network.
        :param method: The ODE solver method.
        :param n_step: The number of integration steps.
        :param atol: Absolute tolerance of the ODE solver.
        :param rtol: Relative tolerance of the ODE solver.
        :param start: Start of the integration interval.
        :param end: End of the integration interval.
        :param return_dlogp: If True, return the dlogp ODE as well.
        :return: None
        """
        self.ode_wrapper = ODEWrapper(b, return_dlogp=return_dlogp)
        self.start, self.end = start, end
        self.rtol, self.atol = rtol, atol
        self.n_step = n_step
        self.method = method
        self.return_dlogp = return_dlogp

    def rollout(self, x0s, beta0s, beta1s) -> tuple:
        """
        Solve the sample and dlogp ODEs given a batch of initial values.
        :param batch: The batch of molecular data.
        :return: The sample and dlogp ODEs.
        """

        batch_size, _ = x0s.shape

        dlogp = torch.zeros(batch_size, 1).to(x0s.device) if self.return_dlogp else None

        if self.return_dlogp:
            states = (x0s, dlogp)

            integration_times = torch.linspace(self.start, self.end, self.n_step).to(x0s.device)

            x, dlogp = odeint_adjoint(func=lambda t, states: self.ode_wrapper(t, states, x0s, beta0s, beta1s),    # might exist be a better solution than lambda function, but works for now!
                                      y0=states,
                                      t=integration_times,
                                      method=self.method,
                                      atol=[self.atol]*len(states),
                                      rtol=[self.rtol]*len(states),
                                      adjoint_params=(self.ode_wrapper.parameters()))
        else:
            states = (x0s,)
            integration_times = torch.linspace(self.start, self.end, self.n_step).to(x0s.device)

            x = odeint_adjoint(func=lambda t, states: self.ode_wrapper(t, states, x0s, beta0s, beta1s),    # might exist be a better solution than lambda function, but works for now!
                               y0=states,
                               t=integration_times,
                               method=self.method,
                               atol=[self.atol],
                               rtol=[self.rtol],
                               adjoint_params=(self.ode_wrapper.parameters()))
            
        return x, dlogp*1e2

