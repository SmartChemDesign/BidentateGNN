#!/bin/sh
#SBATCH --ntasks-per-node=60
#SBATCH --gres=shard:4

python3 Experiments/logK/train.py