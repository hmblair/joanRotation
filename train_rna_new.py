#!/usr/bin/env python3
from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils import data as data
import wandb
from tqdm import tqdm
from bioeq.modules import EquivariantTransformer
from bioeq.polymer import PolymerDataset, GeometricPolymer, Polymer
from bioeq.geom import Repr
from bioeq.utils import LinearWarmupSqrtDecay
from bioeq.geom import RadialBasisFunctions
from bioeq.seq import  sinusoidal_embedding
import warnings
from bioeq._index import(
    Property,
    Element,
    Residue,
    Adenosine,
    Guanosine,
    Cytidine,
    Uridine,
    Reduction,
    Phosphate,
    Ribose
)
import matplotlib.pyplot as plt
import os
import shutil
import dgl

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# Log in to W&B
wandb.login()
# Initialise a run

# The path to the .nc files containing the QM9 dataset
train_path = '../eternafold_train.nc'
val_path = '../eternafold_val.nc'
test_path = ''
# We will not be stacking inputs into batches
batch_size = None
# The number of epochs we will train for
num_epochs = 50
# In case we want to limit the amount of data that the model
# sees. Set to inifinity by default.
MAX_STEP = float('inf')
# The learning rate used by the optimiser
lr = 0.01
# Which variables to load from the datasets (EDIT HERE)
atom_features = ['elements','residues']
residue_features = ['reactivity']
chain_features = []
molecule_features = []
edge_features = []


EDGES_PER_ATOM=16
EDGE_DIM=16

# The datasets which will load the data from the files
train_dataset = PolymerDataset(
    train_path,
    'cpu',
    atom_features,
    residue_features,
    chain_features,
    molecule_features,
    edge_features,
)
val_dataset = PolymerDataset(
    val_path,
    'cpu',
    atom_features,
    residue_features,
    chain_features,
    molecule_features,
    edge_features,
)
MODEL_TYPE = 'geometric'
# The dataloaders which allows us to loop over the files.
train_dataloader = data.DataLoader(train_dataset, batch_size, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size)
HIDDEN_SIZE = 16

hidden_layers = 2
edge_hidden_dim = 32

dropout = 0.3
nheads = 1

name_ix = torch.tensor([
    Adenosine.N9.value,
    Guanosine.N9.value,
    Cytidine.N1.value,
    Uridine.N1.value,
    Phosphate.P.value,
    Ribose.C5p.value,
]).to('cpu')

class GeometricModelRNA(nn.Module):

    def __init__(
        self: GeometricModelRNA,
    ) -> None:
        super().__init__()
        self.elem_embedding = nn.Embedding(5, HIDDEN_SIZE).to(device)
        self.res_embedding = nn.Embedding(4, HIDDEN_SIZE).to(device)
        self.pos_embedding = nn.Linear(1, HIDDEN_SIZE).to(device)
        # Hyperparameters for the model. You will need to change these depending on
        # how you process the inputs

        self.rbf = RadialBasisFunctions(EDGE_DIM).to(device)
        in_repr = Repr([0], 3 * HIDDEN_SIZE)
        out_repr = Repr([0], 2)
        hidden_repr = Repr([0, 1], 3 * HIDDEN_SIZE)
        # Initialise the equiviariant model
        self.model = EquivariantTransformer(
            in_repr,
            out_repr,
            hidden_repr,
            hidden_layers,
            EDGE_DIM,
            edge_hidden_dim,
        ).to(device)

    def forward(
        self: GeometricModelRNA,
        polymer: Polymer,
        elements: torch.Tensor,
        residues: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:

        # Process the inputs to get your node and edge features (EDIT HERE)
        elements = self.elem_embedding(elements.long())
        residues = self.res_embedding(residues.long())
        positions = self.pos_embedding(positions[...,None])
        #charges = charge_embedding(charge[..., None])
        node_features = torch.cat([elements,residues,positions], dim=-1)
        polymer.connect(EDGES_PER_ATOM)
        edge_features = self.rbf(polymer.pdist())
        node_features = node_features[..., None]
        abspos = sinusoidal_embedding(
            tuple(),
            polymer.coordinates.shape[0],
            3*HIDDEN_SIZE,
            device,
        ).view(polymer.num_atoms, 3*HIDDEN_SIZE, 1)
        node_features = node_features + abspos

        # Construct the geometric polymer
        geom_polymer = GeometricPolymer(
            polymer,
            node_features,
            edge_features,
        )

        # Update the polymer using the model
        out_polymer = self.model.polymer(geom_polymer)
        # Sum over each molecule to get a single value per molecule
        p_reactivity = polymer.reduce(
            out_polymer.node_features,
            Property.RESIDUE
        )
        return p_reactivity

class LSTMRNA(nn.Module):

    def __init__(
        self: LSTMRNA,
    ) -> None:
        super().__init__()
        self.model = nn.LSTM(
            input_size=4*HIDDEN_SIZE,
            hidden_size=3*HIDDEN_SIZE,
            num_layers=4,
            batch_first=True,
            bidirectional=True,
        ).to(device)
        self.elem_embedding = nn.Embedding(5, HIDDEN_SIZE).to(device)
        self.res_embedding = nn.Embedding(4, HIDDEN_SIZE).to(device)
        self.coord_embedding = nn.Linear(3, HIDDEN_SIZE).to(device)
        self.projection = nn.Linear(6*HIDDEN_SIZE, 2).to(device)
        self.pos_embedding = nn.Linear(1, HIDDEN_SIZE).to(device)
    def forward(
        self: LSTMRNA,
        polymer: Polymer,
        elements: torch.Tensor,
        residues: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:

        polymer = polymer.center()
        coord_emb = self.coord_embedding(polymer.coordinates)
        elements_emb = self.elem_embedding(elements.long())
        residues_emb = self.res_embedding(residues.long())
        positions_emb = self.pos_embedding(positions[...,None])
        features = torch.cat(
            [coord_emb, elements_emb, residues_emb, positions_emb],
            dim=1
        )

        x, _ = self.model(features)
        x = self.projection(x)

        p_reactivity = polymer.reduce(x,Property.RESIDUE)

        return p_reactivity


if MODEL_TYPE == 'geometric':
    model = GeometricModelRNA()

    run = wandb.init(
    project="GeometricChemicalReactivityModel",
)
elif MODEL_TYPE == 'lstm':
    model = LSTMRNA()
    run = wandb.init(
    project="lstmChemicalReactivityModel",
)
else:
    raise ValueError(f'Invalid model type {MODEL_TYPE}.')

# Initialise the optimiser (We will always use Adam)
optimiser = torch.optim.Adam(
    model.parameters(),
    lr=lr,
)

lr_scheduler = LinearWarmupSqrtDecay(
    optimiser,
    15_000,
)

# For keeping track of the average train and val losses
train_loss_moving_average = 0
val_loss_moving_average = 0
# How often to print the loss
PRINT_LOSS = 10
n = sum(x.numel() for x in model.parameters())
wandb.log({'parameters':n})
shutil.rmtree('geo_with_dist_new2')
os.makedirs('geo_with_dist_new2')
# The main training loop
for epoch in range(num_epochs):

    # For counting which step we are on
    step = 0
    n = 16
    cumulative_loss = 0

    # First, we do the training loop
    model.train()
    for polymer, element, residue, reactivity in tqdm(train_dataloader, desc=f'Epoch {epoch}'):
        missing_nodes = polymer.coordinates.shape[0] - polymer.graph.num_nodes()
        if missing_nodes > 0:
            polymer.graph = dgl.add_nodes(
                polymer.graph,
                missing_nodes
            )
        p1 = polymer.select(torch.tensor([Cytidine.O2.value, Uridine.O2.value, Guanosine.N3.value, Adenosine.N3.value]), Property.NAME)
        p2 = polymer.select(torch.tensor([Ribose.O2p.value]), Property.NAME)
        P_op2_dist = torch.linalg.norm(p1.coordinates - p2.coordinates, dim=-1).to(device)
        polymer = polymer.select(name_ix, Property.NAME).to(device)
        P_op2_dist = P_op2_dist[polymer.indices(Property.RESIDUE)]
        p_reactivity = model(polymer, polymer.elements, polymer.residues, P_op2_dist)
        ix = ~reactivity.isnan()
        reactivity = reactivity[ix].to('cuda:0')
        p_reactivity = p_reactivity[ix]
        # Compute the loss function
        loss = (p_reactivity - reactivity) ** 2
        loss = torch.mean(loss) 

        cumulative_loss += loss
        # Compute the gradients when step is a multiple of n
        if step >0 and step % n == 0:
            mean_loss = cumulative_loss/n
            mean_loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            cumulative_loss = 0
            #g = max(x.grad.abs().max() for x in model.parameters())
            #wandb.log({'max_parameters':g})
        # Log a moving average of the loss every PRINT_LOSS steps
        train_loss_moving_average = train_loss_moving_average * 0.9 + 0.1 * loss.item()
        step += 1
        if step % PRINT_LOSS == 0:
            wandb.log({'Train loss': train_loss_moving_average})
            for x in optimiser.param_groups:
                curr_lr = x['lr']
            wandb.log({'current_learning_rate':curr_lr})
        # If we have reached the maximum step, move on to the next epoch.
        if step > MAX_STEP:
            break

    step = 0
    # Next, we do the validation loop
    #turn off dropout
    model.eval()
    with torch.no_grad():
        for polymer, element, residue, reactivity in tqdm(val_dataloader):
            missing_nodes = polymer.coordinates.shape[0] - polymer.graph.num_nodes()
            if missing_nodes > 0:
                polymer.graph = dgl.add_nodes(
                    polymer.graph,
                    missing_nodes
                )
            p1 = polymer.select(torch.tensor([Cytidine.O2.value, Uridine.O2.value, Guanosine.N3.value, Adenosine.N3.value]), Property.NAME)
            p2 = polymer.select(torch.tensor([Ribose.O2p.value]), Property.NAME)
            P_op2_dist = torch.linalg.norm(p1.coordinates - p2.coordinates, dim=-1).to(device)
            polymer = polymer.select(name_ix, Property.NAME).to(device)
            P_op2_dist = P_op2_dist[polymer.indices(Property.RESIDUE)]
            p_reactivity = model(polymer, polymer.elements, polymer.residues, P_op2_dist)
            ix = ~reactivity.isnan()
            reactivity = reactivity[ix].to('cuda:0')
            p_reactivity = p_reactivity[ix]
            # Compute the loss function
            loss = (p_reactivity - reactivity) ** 2
            loss = torch.mean(loss)
            # Do not compute the gradients in the validation loop
            # since we are not training

            # Log a moving average of the loss every PRINT_LOSS steps
            val_loss_moving_average = val_loss_moving_average * 0.9 + 0.1 * loss.item()
            step += 1
            if step % PRINT_LOSS == 0:
                wandb.log({'Val loss': val_loss_moving_average})
                plt.figure()
                plt.plot(reactivity.cpu().numpy())
                plt.plot(p_reactivity.detach().cpu().numpy())
                plt.legend(['Experimental', 'Predicted'])
                plt.savefig(f'geo_with_dist_new2/reactivity_epoch_{epoch}_step_{step}.png', dpi=300)
                plt.close()
            # If we have reached the maximum step, move on to the next epoch.
            if step > MAX_STEP:
                break
