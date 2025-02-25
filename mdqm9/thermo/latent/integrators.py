import torch
import torch_geometric
from torchdiffeq import odeint_adjoint

from thermo.latent.models.ode_wrapper import ODEWrapper


class MoleculeIntegrator:
    """## Integrator for molecular data. Wraps around the torchdiffeq odeint_adjoint function to allow for PyTorch Geometric molecular data.
    """

    def __init__(self, b: torch.nn.Module, method: str='dopri5', n_step: int=100, atol: float=1e-4, rtol: float=1e-4, 
                 start: float=0.0, end: float=1.0, return_dlogp=False, reverse_ode=False) -> None:
        """## Initialize the integrator.

        ### Args:
            - `b (torch.nn.Module)`: trained velocity model. 
            - `method (str, optional)`: which integration method to use. Defaults to 'dopri5'.
            - `n_step (int, optional)`: number of integration steps. Defaults to 100.
            - `atol (float, optional)`: absolute tolerance. Defaults to 1e-4.
            - `rtol (float, optional)`: relative tolerance. Defaults to 1e-4.
            - `start (float, optional)`: start time. Defaults to 0.0.
            - `end (float, optional)`: end time. Defaults to 1.0.
            - `return_dlogp (bool, optional)`: whether to return the change in log probability. Defaults to False.
            - `reverse_ode (bool, optional)`: whether to integrate in reverse (i.e. from end to start). Defaults to False.
        """

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
        """## Solve the ODEs given a batch of initial values.

        ### Args:
            - `batch (torch_geometric.data.Batch)`: batch of initial values.

        ### Returns:
            - `tuple`: numerical solution to the ODE (position and, if requested, dlogp), dlogp. To resolve indexing, the batch.batch element is also returned. 
        """

        x0 = torch.stack([data.x0 for data in batch.to_data_list()]).unsqueeze(-1)
        batch_size, _, _, _ = x0.shape

        dlogp = torch.zeros(batch_size).to(x0.device)
        x0 = batch.x0.clone()

        if self.return_dlogp:
            states = (x0, dlogp)

            # change the integration times based on the direction of the ODE
            if self.reverse_ode:
                integration_times = torch.linspace(self.end, self.start, self.n_step).to(x0.device)
            else:
                integration_times = torch.linspace(self.start, self.end, self.n_step).to(x0.device)

            xts, dlogp = odeint_adjoint(
                func=lambda t, states: self.ode_wrapper(t, states, batch),    # might exist be a better solution than lambda function, but works for now!
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
                func=lambda t, states: self.ode_wrapper(t, states, batch),
                y0=states,
                t=integration_times,
                method=self.method,
                atol=[self.atol],
                rtol=[self.rtol],
                adjoint_params=(self.ode_wrapper.parameters())
            )
            
        return xts, dlogp, batch.batch

