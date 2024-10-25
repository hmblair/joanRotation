import os
import xarray as xr
from parse import parse_xyz
from tqdm import tqdm
import numpy as np

# Get the path that this index corresponds to
paths = []
folder = 'QM9_data'
for file in os.listdir(folder):
            if file.endswith('xyz'):
                paths.append(folder + '/' + file)
# Load the coordinates and elements of the atom
energy_list = []
for path in tqdm(paths):
    coordinates, elements, energy, charges = parse_xyz(path)
    energy_list.append(energy)
energy_arr = np.array(energy_list).astype(np.float32) 
ds = xr.Dataset({"energy": (["molecule"], energy_arr)})
ds.to_netcdf("energy.nc")
ds2 = xr.load_dataset("energy.nc")
#breakpoint()
