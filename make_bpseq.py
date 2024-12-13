
import xarray as xr
import numpy as np
from tqdm import tqdm
import os
import shutil

IX_TO_RNA = 'ACGU'

file = 'ribonanza1.nc'
ds = xr.open_dataset(file)

reads = ds['reads'].values[:, 0]
ix = np.flip(np.argsort(reads))

ds = ds.isel({'batch': ix})

n = 100_000
ds = ds.isel({'batch': slice(0, n)})

sequences = ds['sequence'].values
sequences_txt = [''.join([IX_TO_RNA[i] for i in sequence])
                 for sequence in sequences]
reactivities = ds['reactivity'].values
reactivities = np.nan_to_num(reactivities, nan=0)

i = 0
for sequence, reactivity in tqdm(zip(sequences_txt, reactivities), total=len(sequences_txt), desc="Processing Sequences"):
    os.makedirs(f'bpseq_shape/{i}')
    for index, base in zip(range(1,178), sequence):
        ind_0 = index - 1
        with open (f'bpseq_shape/{i}/{i}.bpseq', 'a') as f:
            f.write(f'{index} {base} e1 {reactivity[ind_0][0]}\n')
    os.system(f'/scratch/groups/rhiju/hmblair/farfar2/EternaFold/src/contrafold predict bpseq_shape/{i}/{i}.bpseq --evidence --numdatasources 1 --kappa 0.1 --params /scratch/groups/rhiju/hmblair/farfar2/EternaFold/parameters/EternaFoldParams_PLUS_POTENTIALS.v1 --parens bpseq_shape/{i}/{i}.parens')
    with open(f'bpseq_shape/{i}/{i}.parens', 'r') as p:
        lines = p.readlines()
        for num, line in enumerate(lines):
            line = line.strip()
            if line.startswith(">structure"):
                structure = lines[num + 1].strip()
                with open (f'bpseq_shape/{i}/{i}.db','w') as d:
                    d.write(f"{structure}") 
            elif line.startswith(">bpseq_data"):
                sequence = lines[num + 1].strip()
                with open(f'bpseq_shape/{i}/{i}.fasta', 'w') as s:
                    s.write(f">sequence A:1-177\n{sequence}\n")
    i += 1
