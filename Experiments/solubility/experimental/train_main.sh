#!/bin/bash

python Experiments/solubility/experimental/train_main.py \
  --train-sdf "Data/solubility/bigsoldb_train.sdf" \
  --test-sdf "Data/solubility/bigsoldb_test.sdf" \
  --experiment-name "no_MACCSKeys" \
  --folds 5 \
  --seed 23 \
  --batch-size 20 \
  --epochs 1000 \
  --es-patience 100 \
  --mode "regression" \
  --use-exp-values MW RI eps BP d MP WS FP
  # --use-maccskeys \