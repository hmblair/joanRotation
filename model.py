
from __future__ import annotations
import torch
import torch.nn as nn
from feedforward import FeedforwardNetwork

#device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

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
<<<<<<< HEAD
        
        # Initialise the weights of the model to be random
        self.weights = torch.nn.Parameter(
            torch.randn(max_element * (max_element+1) // 2),
            requires_grad=True
        )
        if bias:
            self.bias = torch.nn.Parameter(
                torch.randn(1),
                requires_grad=True
            )
        else:
            self.bias = torch.tensor(0)
        #self.bias.to(device)
=======

        # Get the sizes of the network
        sizes = [3] + hidden_sizes + [3]
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
>>>>>>> 62a1575a6adc67709bb119596f760e98abfdef29

    def forward(
        self: SimpleEnergyModel,
        coordinates: torch.Tensor,
        atom_ix: torch.Tensor,
        charges: torch.Tensor,
        frequencies: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the energy for the input configuartion of atoms.
        """
        # move tensor to GPU
        #coordinates.to(device)
        #atom_ix.to(device) 
        
        # Compute the reciprocal pairwise distances, and
        # set the nan's (along the diagonal) to 0
<<<<<<< HEAD
        d = torch.norm(coordinates[None, :] - coordinates[:, None])       
        reciprocal_d = torch.nan_to_num(1/d, 0.0)
        #reciprocal_d.to(device) 
=======
        d = torch.norm(coordinates[None, :] - coordinates[:, None], dim=-1)
        reciprocal_d = torch.nan_to_num(1/d, 0.0)
        # Limit the distance to 0.1 A
        reciprocal_d[reciprocal_d > 10] = 10
>>>>>>> 62a1575a6adc67709bb119596f760e98abfdef29

        # Map each pair of elements to a unique weight
        pairwise_atom_ix = atom_ix[:,  None] * \
            (atom_ix[:, None] + 1) // 2 + atom_ix[None, :]
<<<<<<< HEAD
        #pairwise_atom_ix.to(device)

        atom_weights = self.weights[pairwise_atom_ix]
        #atom_weights.to(device)
                       
        # Compute the predicted energy under the model
        return COULOMB_CONSTANT * (atom_weights * reciprocal_d).sum() + self.bias
=======
        pairwise_atom_emb = self.embedding(pairwise_atom_ix).squeeze(-1)
        # Get the pairwise product of the charges
        pairwise_charges = charges[:, None] * charges[None, :]
        # Get the pairwise sum of the frequencies
        frequencies = frequencies[:, None] + frequencies[None, :]

        model_inputs = torch.stack(
            [pairwise_atom_emb, pairwise_charges, frequencies],
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
>>>>>>> 62a1575a6adc67709bb119596f760e98abfdef29
