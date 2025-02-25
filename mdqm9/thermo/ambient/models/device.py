import torch


class Module(torch.nn.Module):
    """## Module with defined device
    """

    @property
    def device(self) -> torch.device:
        """## Get device of the module

        ### Returns:
            - `torch.device`: Device of the module
        """
        return next(self.parameters()).device


class DeviceTracker(Module):
    """## Device tracker
    """
    def __init__(self) -> None:
        """### Initialize device tracker
        """
        
        super().__init__()
        self.device_tracker = torch.nn.Parameter(torch.tensor(1.0))
