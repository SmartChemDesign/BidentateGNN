#!/bin/bash

python Experiments/solubility/theoretical/train_main.py \
  --train-sdf "Data/solubility/bigsoldb_train.sdf" \
  --test-sdf "Data/solubility/bigsoldb_test.sdf" \
  --experiment-name "main_no_BP_solvs" \
  --folds 5 \
  --seed 23 \
  --batch-size 20 \
  --epochs 1000 \
  --es-patience 100 \
  --mode "regression" \
  --include-descriptors "eps BP_mols dG"