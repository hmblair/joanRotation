
from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils import data as data
from dataloader import QM9Dataset
from model import SimpleEnergyModel
from constants import NUM_ELEMENTS, ELEMENT_IX
import wandb
from tqdm import tqdm
from timer import EpochTimer
from time import time
import psutil
from bioeq.polymer import GeometricPolymer
from bioeq.geom import Repr
from bioeq.modules import EquivariantTransformer

#device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#CPU memory_usage
process = psutil.Process()

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
num_epochs = 25
# In case we want to limit the amount of data that the model
# sees. Set to inifinity by default.
MAX_STEP = float('inf')
# The learning rate used by the optimiser
lr = 0.001
# The datasets which will load the data from the files
train_dataset = PolymerDataset(train_path, 16)
val_dataset = PolymerDataset(val_path, 16)

# The dataloaders which allows us to loop over the files.
train_dataloader = data.DataLoader(train_dataset, batch_size)
val_dataloader = data.DataLoader(val_dataset, batch_size)

# Initialise the model with NUM_ELEMENTS as the max_elements
in_repr = Repr([0], 1)
out_repr = Repr([1], 4)
hidden_repr = Repr([0, 1], 16)
edge_dim = 1
edge_hidden_dim = 16
nlayers = 2

transformer = EquivariantTransformer(in_repr,
   out_repr,
   hidden_repr,
   nlayers,
   edge_dim,
   edge_hidden_dim,
)
# Initialise the optimiser (We will always use Adam)
optimiser = torch.optim.Adam(model.parameters(), lr=lr)

# For keeping track of the average train and val losses
train_loss_moving_average = 0
val_loss_moving_average = 0
# How often to print the loss
PRINT_LOSS = 1_000

# The main training loop
for epoch in tqdm(range(num_epochs), desc='Current Epoch:'):

    # start gpu timer
    #timer = EpochTimer()
    time1 = time() #call cpu time

    # For counting which step we are on
    step = 0

    # First, we do the training loop
    #for coordinates, atoms, energy, charges in train_dataloader: 
    for polymer in train_dataloader:
        polymer.node_features = polymer.node_features[...,None].to(torch.float32)
        out_polymer = transformer(polymer)    
        # Compute the predicted energy using the model
        #p_energy = model(coordinates, atoms, charges)
        #p_energy.to(device)

        # Compute the loss function
        #loss = (p_energy - energy) ** 2

        # Compute the gradients
        #loss.backward()

        # Use the gradients to update the weights
        #optimiser.step()

        # Set the gradients back to zero
        #optimiser.zero_grad()

        # Log a moving average of the loss every PRINT_LOSS steps
        #train_loss_moving_average = train_loss_moving_average * 0.9 + 0.1 * loss
        step += 1
        #if step % PRINT_LOSS == 0:
            #wandb.log({'Train loss': train_loss_moving_average})
            #wandb.log({'Memory usage':process.memory_info().rss}) 

        # If we have reached the maximum step, move on to the next epoch.
        if step > MAX_STEP:
            break
         
    train_dataset.shuffle()
    # Next, we do the validation loop
    #for coordinates, atoms, energy, charges in val_dataloader:
     for polymer in val_dataloader:
        polymer.node_features = polymer.node_features[...,None].to(torch.float32)
        out_polymer = transformer.polymer(polymer)
        # Compute the predicted energy using the model
        #p_energy = model(coordinates, atoms, charges)

        # Compute the loss function
        #loss = (p_energy - energy) ** 2

        # Do not compute the gradients in the validation loop
        # since we are not training

        # Log a moving average of the loss every PRINT_LOSS steps
        #val_loss_moving_average = val_loss_moving_average * 0.9 + 0.1 * loss
        step += 1
        #if step % PRINT_LOSS == 0:
            #wandb.log({'Val loss': val_loss_moving_average})
            #wandb.log({'Memory usage': process.memory_info().rss})
    
# end gpu timer
    #epoch_time = timer.finish()
    #print(epoch_time)
    time2 = time() #call cpu time again
    epoch_cpu_time = time2 - time1
    print (epoch_cpu_time)
