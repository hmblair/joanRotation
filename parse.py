import numpy as np
from biotite.structure.io import load_structure
import biotite
import torch
import dgl

INTERNAL_ENERGY_IX = 10
FREE_ENERGY_IX = 13


def parse_float(s: str) -> float:
    """
    Parse a float which is potentially in scientific notation.
    """
    try:
        return float(s)
    except ValueError:
        base, power = s.split('*^')
        return float(base) * 10**float(power)


def parse_xyz(filename: str) -> tuple[np.ndarray, list[str], float]:
    """
    Parses QM9 specific xyz files.
    See https://www.nature.com/articles/sdata201422/tables/2 for reference.
    """
    num_atoms = 0
    elements = []
    coordinates = []
    charges = []
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f):
            if line_num == 0:
                num_atoms = int(line)
            elif line_num == 1:
                ls = line.split()[2:]
                energy = parse_float(ls[INTERNAL_ENERGY_IX])
            elif 2 <= line_num <= 1 + num_atoms:
                element, x, y, z, charge = line.split()
                elements.append(element)
                coordinates.append(
                    [parse_float(x), parse_float(y), parse_float(z)]
                )
                charges.append(parse_float(charge))

    coordinates = np.array(coordinates)

    return coordinates, elements, energy, charges

def parse_pdb(filename: str) -> tensor:
    """
    Parses bond_list from pdb files. 
    Creates 2 tensors: origin -> destination and destination -> origin (covers both directions)
    """
    with open(filename, 'r') as f:
        pdb = load_structure(f)
        bond_list = biotite.structure.connect_via_distances(pdb)
        bond_array = bond_list.as_array()
        bond_array = bond_array.astype(np.int64)
        bond_tensor = torch.from_numpy(bond_array)
        bond_origin = bond_tensor[:, 0]
        bond_destination = bond_tensor[:, 1]
        U = torch.cat((bond_origin,bond_destination))
        V = torch.cat((bond_destination,bond_origin))
    return U, V