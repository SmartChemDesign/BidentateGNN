#!/bin/sh
#SBATCH --ntasks-per-node=30
#SBATCH --gres=shard:4

python3 Experiments/redox/GCNN_FCNN/sample_training/exp_only/train.py