
from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils import data as data
from dataloader import QM9Dataset
from model import SimpleEnergyModel
from constants import NUM_ELEMENTS, ELEMENT_IX
import wandb
from tqdm import tqdm

# Log in to W&B
wandb.login()
# Initialise a run
run = wandb.init(
    # Set the project where this run will be logged
    project="SimpleEnergyModel",
    # Track hyperparameters and run metadata
)

# The path to the folders containing the QM9 dataset
train_path = 'train'
val_path = 'validation'
test_path = 'test'
# We will not be stacking inputs into batches
batch_size = None
# The number of epochs we will train for
num_epochs = 5
# In case we want to limit the amount of data that the model
# sees. Set to inifinity by default.
MAX_STEP = float('inf')
# The learning rate used by the optimiser
lr = 0.001
# The datasets which will load the data from the files
train_dataset = QM9Dataset(train_path)
val_dataset = QM9Dataset(val_path)
# The dataloaders which allows us to loop over the files.
train_dataloader = data.DataLoader(train_dataset, batch_size)
val_dataloader = data.DataLoader(val_dataset, batch_size)

# Initialise the model with NUM_ELEMENTS as the max_elements
model = SimpleEnergyModel(NUM_ELEMENTS)
# Initialise the optimiser (We will always use Adam)
optimiser = torch.optim.Adam(model.parameters(), lr=lr)

# For keeping track of the average train and val losses
train_loss_moving_average = 0
val_loss_moving_average = 0
# How often to print the loss
PRINT_LOSS = 1_000

# The main training loop
for epoch in tqdm(range(num_epochs), desc='Current Epoch:'):

    # For counting which step we are on
    step = 0

    # First, we do the training loop
    for coordinates, atoms, energy in train_dataloader:

        # Compute the predicted energy using the model
        p_energy = model(coordinates, atoms)

        # Compute the loss function
        loss = (p_energy - energy) ** 2

        # Compute the gradients
        loss.backward()

        # Use the gradients to update the weights
        optimiser.step()

        # Set the gradients back to zero
        optimiser.zero_grad()

        # Log a moving average of the loss every PRINT_LOSS steps
        train_loss_moving_average = train_loss_moving_average * 0.9 + 0.1 * loss
        step += 1
        if step % PRINT_LOSS == 0:
            wandb.log({'Train loss': train_loss_moving_average})

        # If we have reached the maximum step, move on to the next epoch.
        if step > MAX_STEP:
            break

    # Next, we do the validation loop
    for coordinates, atoms, energy in val_dataloader:

        # Compute the predicted energy using the model
        p_energy = model(coordinates, atoms)

        # Compute the loss function
        loss = (p_energy - energy) ** 2

        # Do not compute the gradients in the validation loop
        # since we are not training

        # Log a moving average of the loss every PRINT_LOSS steps
        val_loss_moving_average = val_loss_moving_average * 0.9 + 0.1 * loss
        step += 1
        if step % PRINT_LOSS == 0:
            wandb.log({'Val loss': val_loss_moving_average})
