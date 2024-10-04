
import torch
import torch.nn as nn


class FeedforwardLayer(nn.Module):

    def __init__(
        self,
        in_size,
        out_size,
        activation: bool = True,
    ) -> None:

        self.linear = nn.Linear(
            in_size,
            out_size
        )
        self.activation = nn.ReLU()
        self.use_activation = activation

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.linear(x)
        if self.use_activation:
            x = self.activation(x)
        return x


class FeedforwardNetwork(nn.Module):

    def __init__(
        self,
        dimensions: list[int],
    ) -> None:

        # Initialise the layers using the provided dimensions
        # Use no activation for the final layer.
        layers = []

        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # Apply the layers in the order they are given in the list.
        pass


# Initialise an instance of a FeedforwardNetwork using the given dimensions
dimensions = [3, 10, 10, 3]
network = ...

# Pass the given input to the network.
x = torch.randn(3)
