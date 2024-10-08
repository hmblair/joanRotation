
import torch
import torch.nn as nn


class FeedforwardLayer(nn.Module):

    def __init__(
        self,
        in_size,
        out_size,
        activation: bool = True,
    ) -> None:
        super().__init__()
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
        super().__init__()
        # Initialise the layers using the provided dimensions
        # Use no activation for the final layer.
        layers = []
        if len(dimensions) >= 2:
            for i in range(len(dimensions)-2):
                layer = FeedforwardLayer(dimensions[i], dimensions[i+1], True)
                layers.append(layer)
            layers.append(FeedforwardLayer(dimensions[-2], dimensions[-1], False))
    
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # Apply the layers in the order they are given in the list.
        for layer in self.layers:
            x = layer(x)
        return x



# Initialise an instance of a FeedforwardNetwork using the given dimensions
dimensions = [7, 9, 5, 8, 6]
network = FeedforwardNetwork(dimensions)
# Pass the given input to the network.
x = torch.randn(7)
y = network(x)
print(x.shape)
print(y.shape)