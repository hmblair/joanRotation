from parse import parse_xyz

# Get the path that this index corresponds to
path = 'QM9_dataset'
for file in os.listdir(folder):
            if file.endswith('xyz'):
                self.paths.append(folder + '/' + file)
# Load the coordinates and elements of the atom
coordinates, elements, energy, charges = parse_xyz(path)
# Convert the energy to a PyTorch tensor
energy_arr = np.array(energy) 
ds = xarray.Dataset({‘energy’: ([‘molecule’], energy_arr)})
ds.to_netcdf(‘energy.nc’)
xr.load_dataset(‘energy.nc’)

