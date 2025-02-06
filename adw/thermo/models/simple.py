import torch
import torch.nn as nn


class FCNetMultiBeta(nn.Module):

    """
    Fully connected neural network taking beta0 and beta1 as input.
    """

    def __init__(self, in_size, out_size, hidden_size, num_layers):
        """
        Initialize fully connected neural network.
        :param in_size: int, input size
        :param out_size: int, output size
        :param hidden_size: int, hidden size
        :param num_layers: int, number of hidden layers
        """
        super(FCNetMultiBeta, self).__init__()

        # define layers
        sizes = [in_size + 2] + [hidden_size] * num_layers + [out_size]
        layers = []

        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))

            if i != len(sizes) - 2:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

        self.beta_embed = nn.Sequential(nn.Linear(3, hidden_size),
                                        nn.SiLU(),
                                        nn.Linear(hidden_size, hidden_size),
                                        nn.SiLU(),
                                        nn.Linear(hidden_size, 1))

    def forward(self, x0s, xts, ts, beta0s, beta1s):
        beta_embed = self.beta_embed(torch.cat([beta0s, beta1s, ts], dim=1))
        model_input = torch.cat([xts, ts, beta_embed], dim=1)
        return self.net(model_input)


