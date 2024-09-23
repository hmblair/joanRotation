
from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils import data as data
from dataloader import QM9Dataset
from model import SimpleEnergyModel
from constants import NUM_ELEMENTS, ELEMENT_IX

# The device we will be using for training
device = 'cpu'
# The path to the folder containing the QM9 dataset
path = 'data'
# We will not be stacking inputs into batches
batch_size = None
# The number of epochs we will train for
num_epochs = 5
# The learning rate used by the optimiser
lr = 0.001
# The dataset which will load the data from the files
dataset = QM9Dataset(path)
# The dataloader which allows us to loop over the files.
dataloader = data.DataLoader(dataset, batch_size)

# Initialise the model
model = SimpleEnergyModel(NUM_ELEMENTS)
# Move the model to the correct device
model = model.to(device)
# Initialise the optimiser
optimiser = torch.optim.Adam(model.parameters(), lr=lr)

step = 0
loss_moving_average = 0
for epoch in range(num_epochs):
    for coordinates, atoms, energy in dataloader:

        atom_ix = torch.Tensor([ELEMENT_IX[atom] for atom in atoms]).long()

        # Move the inputs and outputs to the appropriate device
        coordinates = coordinates.to(device)
        atom_ix = atom_ix.to(device)
        energy = energy.to(device)

        # Compute the predicted energy using the model
        p_energy = model(coordinates, atom_ix)

        # Compute the loss function
        loss = (energy - p_energy) ** 2

        # Compute the gradients
        loss.backward()

        # Perform the backward step
        optimiser.step()

        # Zero the gradients
        optimiser.zero_grad()

        loss_moving_average = loss_moving_average * 0.9 + 0.1 * loss
        step += 1
        if step % 2500 == 0:
            print(f'Loss (moving average): {loss_moving_average.item():.3f}')
