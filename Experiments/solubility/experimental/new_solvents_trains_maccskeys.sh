#!/bin/bash

solvents=("n-pentanol" "THF" "ethyl acetate" "2-butanone" "cyclohexane" "n-hexane" "isobutanol" "1,4-dioxane" "DMF" "toluene" "acetonitrile" "water" "methyl acetate" "n-butanol" "n-propanol" "acetone" "isopropanol" "methanol" "ethanol" "n-butyl acetate" "acetic acid" "isopropyl acetate" "n-propyl acetate" "propylene glycol")

for solvent in "${solvents[@]}"; do
  python Experiments/solubility/experimental/train.py \
    --sdf "Data/solubility/bigsoldb_full.sdf" \
    --experiment-name "maccskeys_${solvent}_log_0" \
    --folds 1 \
    --seed 23 \
    --batch-size 500 \
    --epochs 1000 \
    --es-patience 100 \
    --mode "regression" \
    --include-descriptors " " \
    --test-solvents "${solvent}" \
    --use-maccskeys True \
    --n-test-in-train 0

  for i in {1..10}; do
    python Experiments/solubility/experimental/train.py \
      --sdf "Data/solubility/bigsoldb_full.sdf" \
      --experiment-name "maccskeys_${solvent}_log_10_${i}" \
      --folds 1 \
      --seed 23 \
      --batch-size 500 \
      --epochs 1000 \
      --es-patience 100 \
      --mode "regression" \
      --include-descriptors " " \
      --test-solvents "${solvent}" \
      --use-maccskeys True \
      --n-test-in-train 10
  done

  for i in {1..10}; do
    python Experiments/solubility/experimental/train.py \
      --sdf "Data/solubility/bigsoldb_full.sdf" \
      --experiment-name "maccskeys_${solvent}_log_50_${i}" \
      --folds 1 \
      --seed 23 \
      --batch-size 500 \
      --epochs 1000 \
      --es-patience 100 \
      --mode "regression" \
      --include-descriptors " " \
      --test-solvents "${solvent}" \
      --use-maccskeys True \
      --n-test-in-train 50
  done

done

