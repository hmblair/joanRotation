
from __future__ import annotations
from parse import parse_xyz
import torch
from torch.utils import data
import os
from constants import ELEMENT_IX


class QM9Dataset(data.Dataset):
    """
    For loading batches of data from the QM9 dataset.
    """

    def __init__(
        self: QM9Dataset,
        folder: str,
    ) -> None:

        # Get the paths of all the files in the data folder
        self.paths = []
        for file in os.listdir(folder):
            if file.endswith('xyz'):
                self.paths.append(folder + '/' + file)

    def __getitem__(
        self: QM9Dataset,
        ix: int,
    ) -> tuple[torch.Tensor]:
        """
        Get the data from the .xyz file at index ix.
        """

        # Get the path that this index corresponds to
        path = self.paths[ix]
        # Load the coordinates and elements of the atom
        coordinates, elements, energy = parse_xyz(path)
        # Convert the coordinates to a PyTorch tensor
        coordinates = torch.from_numpy(coordinates)
        # Convert the elements into integers
        elements = [ELEMENT_IX[element] for element in elements]
        # Convert the elements to a PyTorch tensor
        elements = torch.tensor(elements) 
        # Convert the energy to a PyTorch tensor
        energy = torch.tensor(energy)

        # Convert the coordinates and energies to
        # standard floats
        coordinates = coordinates.to(torch.float32)
        energy = energy.to(torch.float32)

        return coordinates, elements, energy

    def __len__(
        self: QM9Dataset,
    ) -> int:
        """
        Return the number of datapoints in the dataset.
        """

        return len(self.paths)
