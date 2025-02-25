import torch
import torch_geometric
from torchdiffeq import odeint_adjoint

from thermo.ambient.models.ode_wrapper import ODEWrapper


class MoleculeIntegrator:
    """## Integrator for molecular data. Wraps around the torchdiffeq odeint_adjoint function to allow for PyTorch Geometric molecular data.
    """

    def __init__(self, b: torch.nn.Module, method: str='dopri5', n_step: int=100, atol: float=1e-4, rtol: float=1e-4, 
                 start: float=0.0, end: float=1.0, return_dlogp: bool=False, reverse_ode: bool=False) -> None:

        self.ode_wrapper = ODEWrapper(  # wrap velocity model into differential equation (includes position and, if to be returned, dlogp)
            b=b, 
            return_dlogp=return_dlogp, 
            reverse_ode=reverse_ode
        )

        self.start, self.end = start, end
        self.rtol, self.atol = rtol, atol
        self.n_step = n_step
        self.method = method
        self.return_dlogp = return_dlogp
        self.reverse_ode = reverse_ode

    def rollout(self, batch: torch_geometric.data.Batch) -> tuple:
        x0 = torch.stack([data.x0 for data in batch.to_data_list()]).unsqueeze(-1)
        batch_size, _, _, _ = x0.shape

        dlogp = torch.zeros(batch_size).to(x0.device)
        x0 = batch.x0.clone()
        n_steps = [0]

        if self.return_dlogp:
            states = (x0, dlogp)

            # change the integration times based on the direction of the ODE
            if self.reverse_ode:
                integration_times = torch.linspace(self.end, self.start, self.n_step).to(x0.device)
            else:
                integration_times = torch.linspace(self.start, self.end, self.n_step).to(x0.device)

            xts, dlogp = odeint_adjoint(
                func=lambda t, states: self.ode_wrapper(t, states, batch, n_steps),  # might exist be a better solution than lambda function, but works for now!
                y0=states,
                t=integration_times,
                method=self.method,
                atol=[self.atol]*len(states),
                rtol=[self.rtol]*len(states),
                adjoint_params=(self.ode_wrapper.parameters())
            )
        else:
            states = (x0)
            integration_times = torch.linspace(self.start, self.end, self.n_step).to(x0.device)

            xts = odeint_adjoint(
                func=lambda t, states: self.ode_wrapper(t, states, batch, n_steps),
                y0=states,
                t=integration_times,
                method=self.method,
                atol=[self.atol],
                rtol=[self.rtol],
                adjoint_params=(self.ode_wrapper.parameters())
            )
            
        return xts, dlogp*1e2, len(n_steps)-1, batch.batch

