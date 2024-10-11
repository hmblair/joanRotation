
from __future__ import annotations
from parse import parse_xyz
from parse import parse_pdb
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
 
class pdbDataset(data.Dataset):
    """
    For loading batches of data from pdb files.
    """

    def __init__(
        self: pdbDataset,
        folder: str,
    ) -> None:

        # Get the paths of all the files in the data folder
        self.paths = []
        for file in os.listdir(folder):
            if file.endswith('pdb'):
                self.paths.append(folder + '/' + file)

    def __getitem__(
        self: pdbDataset,
        ix: int,
    ) -> tuple[torch.Tensor]:
        """
        Get the data from the .pdb file at index ix.
        """

        # Get the path that this index corresponds to
        path = self.paths[ix]
        # Parse bond tensors
        U, V = parse_pdb(path)
        # Graph bonds as edges
        graph = dgl.graph((U,V))
        # Add self <-> self bonds to the graph
        graph = dgl.add_self_loop(graph)
        return graph
        
    
#dataset = QM9Dataset('QM9_data')
#coordinates, elements, energy = dataset[0]
#print(coordinates, elements, energy, len(dataset))
