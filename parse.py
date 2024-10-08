
import numpy as np

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
    harmonic_vibrational_frequencies = []
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
            elif line_num == num_atoms + 2:
                harmonic_vibrational_frequencies = [float(i) for i in line.split()]

    coordinates = np.array(coordinates)

    return coordinates, elements, energy, charges, harmonic_vibrational_frequencies
