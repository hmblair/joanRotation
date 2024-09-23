
from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils import data as data
from dataloader import QM9Dataset
from model import SimpleEnergyModel
from constants import NUM_ELEMENTS, ELEMENT_IX

# The path to the folder containing the QM9 dataset
path = 'data'
# We will not be stacking inputs into batches
batch_size = None
# The number of epochs we will train for
num_epochs = 5
# The learning rate used by the optimiser
lr = 0.001
# The dataset which will load the data from the files
dataset = ...
# The dataloader which allows us to loop over the files.
dataloader = data.DataLoader(dataset, batch_size)

# Initialise the model with NUM_ELEMENTS as the max_elements
model = ...
# Initialise the optimiser (We will always use Adam)
optimiser = torch.optim.Adam(model.parameters(), lr=lr)

# For counting which step we are on
step = 0
# For keeping track of the average loss
loss_moving_average = 0
# How often to print the loss
PRINT_LOSS = 2500

# The main training loop
for epoch in range(num_epochs):
    # Loop over the dataset
    for coordinates, atoms, energy in dataloader:

        # Compute the predicted energy using the model
        p_energy = ...

        # Compute the loss function
        loss = ...

        # Compute the gradients
        pass

        # Use the gradients to update the weights
        pass

        # Set the gradients back to zero
        pass

        # Print a moving average of the loss every PRINT_LOSS steps
        loss_moving_average = loss_moving_average * 0.9 + 0.1 * loss
        step += 1
        if step % PRINT_LOSS == 0:
            print(
                f'Loss (moving average; step {step}): {loss_moving_average.item():.3f}')
