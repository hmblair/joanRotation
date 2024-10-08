
from __future__ import annotations
import torch
import torch.nn as nn
from feedforward import FeedforwardNetwork


COULOMB_CONSTANT = -1
HARTREE_CONSTANT = 0.0367
EV = 1.602E-19


class SimpleEnergyModel(nn.Module):
    """
    A simple energy model with trainable weights. 

        max_element: determines the maximum atomic number
        that the model will accept.
        hidden_sizes: the hidden sizes to use for the network which computes the
        Coulomb and Lennard-Jones coefficients.
    """

    def __init__(
        self: SimpleEnergyModel,
        max_element: int,
        hidden_sizes: list[int]
    ) -> None:
        super().__init__()

        # Get the sizes of the network
        sizes = [2] + hidden_sizes + [3]
        # Initialise the network
        self.network = FeedforwardNetwork(sizes)
        # Initialise a 1-dimensional embedding
        # for the pairwise atom types
        self.embedding = nn.Embedding(
            max_element * (max_element + 1) // 2,
            1,
        )
        # Initialise the bias to be random
        self.bias = nn.Parameter(
            torch.randn(1),
            requires_grad=True,
        )

    def forward(
        self: SimpleEnergyModel,
        coordinates: torch.Tensor,
        atom_ix: torch.Tensor,
        charges: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the energy for the input configuartion of atoms.
        """

        # Compute the reciprocal pairwise distances, and
        # set the nan's (along the diagonal) to 0
        d = torch.norm(coordinates[None, :] - coordinates[:, None], dim=-1)
        reciprocal_d = torch.nan_to_num(1/d, 0.0)
        # Limit the distance to 0.1 A
        reciprocal_d[reciprocal_d > 10] = 10

        # Map each pair of elements to a unique weight
        pairwise_atom_ix = atom_ix[:,  None] * \
            (atom_ix[:, None] + 1) // 2 + atom_ix[None, :]
        pairwise_atom_emb = self.embedding(pairwise_atom_ix).squeeze(-1)
        # Get the pairwise product of the charges
        pairwise_charges = charges[:, None] * charges[None, :]

        model_inputs = torch.stack(
            [pairwise_atom_emb, pairwise_charges],
            dim=-1,
        )

        # Apply the model. The first set of weights
        # will be used by the Coulomb model, and the
        # second and third by the Lennard-Jones model.
        atom_weights = self.network(model_inputs)
        c_weights = atom_weights[..., 0]
        lj_sigma = atom_weights[..., 1]
        lj_r = atom_weights[..., 2]

        # Compute the predicted energy under the coulomb model
        c_energy = COULOMB_CONSTANT * (c_weights * reciprocal_d).sum()
        # Compute the predicted energy under the Lennard-Jones model
        lj_energy = EV * (lj_r * ((lj_sigma * reciprocal_d) **
                          12 - (lj_sigma * reciprocal_d) ** 6)).sum()

        # Return the sum of the two model predictions
        # and the bias
        return c_energy + lj_energy + self.bias
