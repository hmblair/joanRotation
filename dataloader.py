
from __future__ import annotations
from parse import parse_xyz
from parse import parse_cif
import torch
from torch.utils import data
import os
from constants import ELEMENT_IX
import random
import numpy as np
from biotite.structure.io import load_structure
import biotite
import torch
import dgl
import xarray as xr

ELEMENT_IX = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20
}

#device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

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
        coordinates, elements, energy, charges = parse_xyz(path)
        # Convert the coordinates to a PyTorch tensor
        coordinates = torch.from_numpy(coordinates)
        # Convert the elements into integers
        elements = [ELEMENT_IX[element] for element in elements]
        # Convert the elements to a PyTorch tensor
        elements = torch.tensor(elements) 
        # Convert the energy to a PyTorch tensor
        energy = torch.tensor(energy)
        # Convert the charges to a PyTorch tensor
        charges = torch.tensor(charges)

        # Convert the coordinates and energies to
        # standard floats
        coordinates = coordinates.to(torch.float32)
        energy = energy.to(torch.float32)
        charges = charges.to(torch.float32)

        # move tensor to GPU
        #coordinates.to(device)
        #elements.to(device)
        #energy.to(device)

        return coordinates, elements, energy, charges

    def __len__(
        self: QM9Dataset,
    ) -> int:
        """
        Return the number of datapoints in the dataset.
        """

        return len(self.paths)

    def shuffle(
        self: QM9Dataset,
    ) -> list:
        """
        shuffle the data/list of paths
        """
        return random.shuffle(self.paths)
 
class CifDataset(data.Dataset):
    """
    For loading batches of data from cif files.
    """

    def __init__(
        self: CifDataset,
        folder: str,
    ) -> None:

        # Get the paths of all the files in the data folder
        self.paths = []
        for file in os.listdir(folder):
            if file.endswith('bcif'):
                self.paths.append(folder + '/' + file)

    def __getitem__(
        self: CifDataset,
        ix: int,
    ) -> tuple[torch.Tensor]:
        """
        Get the data from the .bcif file at index ix.
        """

        # Get the path that this index corresponds to
        path = self.paths[ix]
        # Parse bond tensors and coordinates
        U, V, coordinates = parse_cif(path)
        # Graph bonds as edges
        graph = dgl.graph((U,V))
        # Add self <-> self bonds to the graph
        graph = dgl.add_self_loop(graph)
        # convert the elements into integers
        elements = [ELEMENT_IX[element] for element in elements]
        # Convert the elements to a PyTorch tensor
        elements = torch.tensor(elements)
        return graph, coordinates, elements
        
class EnergyDataset(data.Dataset):
    """
    For loading batches of data from the QM9 dataset.
    """

    def __init__(
        self: EnergyDataset,
        file: str,
    ) -> None:
   
        self.energy_file = xr.load_dataset(file)   
    def __getitem__(
        self: EnergyDataset,
        ix: int
    ):
        energy = self.energy_file['energy'].values[ix] 
        return energy

    def __len__(
        self: EnergyDataset
):
        return self.energy_file.sizes["molecule"]

dataset = EnergyDataset('energy.nc')
print(len(dataset))
print(dataset[5])
#coordinates, elements, energy = dataset[0]
#print(coordinates, elements, energy, len(dataset))
