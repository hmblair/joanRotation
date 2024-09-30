#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --time=04:00:00
#SBATCH --partition=rhiju
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8 
#SBATCH --mem=32G

ml load python/3.12.1
pip install -r requirements.txt

export WANDB_API_KEY=...

python3 train.py
