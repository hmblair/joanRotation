
from __future__ import annotations
import torch
import torch.nn as nn


class SimpleEnergyModel(nn.Module):
    """
    A simple energy model with trainable weights.
    """

    def __init__(
        self: SimpleEnergyModel,
        max_atoms: int,
    ) -> None:
        super().__init__()

        # Initialise the weights of the model to be random
        self.weights = torch.nn.Parameter(
            torch.randn(max_atoms * (max_atoms+1) // 2),
            requires_grad=True
        )
        self.bias = torch.nn.Parameter(
            torch.randn(1),
            requires_grad=True
        )

    def forward(
        self: SimpleEnergyModel,
        coordinates: torch.Tensor,
        atom_ix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the energy for the input configuartion of atoms.
        """

        # Compute the reciprocal pairwise distances, and
        # set the nan's (along the diagonal) to 0
        d = torch.norm(coordinates[None, :] - coordinates[:, None])
        reciprocal_d = torch.nan_to_num(1/d, 0.0)

        # Map each pair of atoms to a unique weight
        pairwise_atom_ix = atom_ix[:,  None] * \
            (atom_ix[:, None] + 1) // 2 + atom_ix[None, :]
        atom_weights = self.weights[pairwise_atom_ix]

        # Compute the predicted energy under the model
        return (atom_weights * reciprocal_d).sum() + self.bias
