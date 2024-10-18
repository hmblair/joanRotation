# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:08:53 2024

@author: Joan Lee
"""

import biotite
from parse import parse_xyz
import os
import numpy as np

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

folder = "QM9_data"
paths = []
for file in os.listdir(folder):
    if file.endswith('xyz'):
        paths.append(folder + '/' + file)

        
for path in paths:
    file_name = path.split("/")
    file_name = file_name[1]
    file_name = file_name.split(".")
    file_name = file_name[0]
    atoms = []
    coordinates, elements, energy, charges = parse_xyz(path)
    num_rows = coordinates.shape[0]
    for i in range(num_rows):
        atom = biotite.structure.Atom(coordinates[i], element = elements[i], charge = charges[i])
        atoms.append(atom)
    atom_array = biotite.structure.array(atoms)
    #atom_array.set_annotation("charge", charges)
    pdb = biotite.structure.io.save_structure(f"QM9_pdbs/{file_name}.pdb", atom_array)
#breakpoint()
