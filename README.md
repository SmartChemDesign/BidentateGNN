# Multi-input graph neural networks for molecular property prediction

Graph neural network models for three molecular property prediction tasks, sharing
a common featurization, training and evaluation codebase:

| Task | Property | Data |
|---|---|---|
| **redox** | Oxidation and reduction potentials of organic molecules in solution | experimental + quantum-chemically calculated |
| **logK** | Stability constants of metal complexes | experimental |
| **solubility** | Decimal logarithm of solubility (logS) | experimental |

All models combine a molecular graph branch (GCNN) with a branch encoding the
chemical environment — solvent descriptors for redox and solubility, metal and
condition descriptors for logK — and are trained with K-fold cross-validation and
a held-out test set.

<!-- TODO: add links to the papers and citation blocks for each task. -->
<!-- TODO: name the license here; the LICENSE file is already in the repository. -->

## Requirements

- Linux with conda ([miniconda](https://docs.conda.io/projects/miniconda/en/latest/))
- Python 3.10

## Installation

```bash
conda env create -f environment.yml
conda activate torch_geometric
```

Verify the installation:

```bash
python -c "import torch, torch_geometric, rdkit; print(torch.__version__, torch.cuda.is_available())"
```

## Repository layout

```
Data/{logK,redox,solubility}/          datasets and reference tables
Experiments/{logK,redox,solubility}/   training and prediction scripts with their configs
Models/{logK,redox,solubility}/        trained models released with the papers
Output/                                results of local runs, created at run time
Source/                                shared library code
    config.py                          project root path
    data.py                            dataset splitting strategies
    experiment_utils.py                experiment logging, split statistics, QC plots
    trainer.py                         cross-validation loop and fold ensembles
    GCNN/, GCNN_FCNN/                  models and featurizers
environment.yml                        pinned conda environment
```

All commands below are run from the repository root.

## Redox potentials

Multifidelity model bridging experimental and calculated redox potentials. The
experimental records form the high-fidelity stream and the quantum-chemically
calculated ones the low-fidelity stream; the two enter a weighted loss so that the
calculated data provides coverage while the experimental data drives model
selection.

### Data

| File | Location |
|---|---|
| `exp_dataset.sdf` | `Data/redox/dataset/exp_only/` |
| `exp_calc_dataset.sdf` | `Data/redox/dataset/calc_exp/` |
| `solvent_properties.csv` | `Data/redox/additional/` |

`solvent_properties.csv` holds 9 physical descriptors for 14 solvents and is the
reference table for solvent names and SMILES.

### Training

```bash
# Graph-only model on experimental data (separate oxidation and reduction models)
python3 Experiments/redox/GCNN/train.py

# Model with a solvent branch, experimental data only
python3 Experiments/redox/GCNN_FCNN/sample_training/exp_only/train.py

# Model with a solvent branch, experimental + calculated data
python3 Experiments/redox/GCNN_FCNN/sample_training/calc_exp/train.py
```

Runs use solvent-stratified K-fold cross-validation with a held-out test set. Every
solvent present in the test set is guaranteed to appear in training. 
All records of a test molecule are removed from the training folds so that no information leaks through the low-fidelity branch.

### Output layout

```
Output/redox/<model>/<experiment>/
    model_structure.json, model_config.torch
    models/fold_<n>/best_model.pt, losses.json
    metrics/fold_<n>_metrics.json, crossval_summary.json, metrics_table.csv,
            ensemble_test_metrics.json, solvent_metrics.csv
    predictions/fold_<n>/{train,val,val_exp}_predictions.csv
    predictions/test_predictions.csv
```

Test metrics are computed for the ensemble of per-fold models, whose prediction is
the mean across folds.

## Stability constants (logK)

```bash
python3 Experiments/logK/train.py       
python3 Experiments/logK/predict.py
```

`tanimoto_split_train.py` trains the same model on a Tanimoto-similarity split,
which separates structurally similar molecules between training and test and gives
a stricter estimate of generalization.

`train.sh` and `predict.sh` are batch submission wrappers for the same scripts.

<!-- TODO: confirm the scheduler these .sh scripts target and document how to
     adapt them (partition, walltime, module loads). -->

## Solubility

Models predict the decimal logarithm of solubility (logS). Two model variants are
provided (use theoretical or experimental descriptors for solvents).

### Data

`bigsoldb_full.sdf`, `bigsoldb_test.sdf`, `bigsoldb_train.sdf`, located in `Data/solubility/`.

`solvent_properties_2.0.csv` is a reference table containing 8 physicochemical characteristics for 31 solvents, used to generate solvent descriptor vectors.

### Theoretical descriptors model

Uses a set of theoretical descriptors. Some of them are optional:
- `eps`: dielectric permittivity
- `BP_mols`: boiling point of molecules
- `BP_solvs`: boiling point of solvents
- `dG`: predicted Gibbs free energy of solvation

#### Training on a pre-split dataset (example with all optional descriptors)

```bash
bash Experiments/solubility/theoretical/train_main_all.sh
```

### Experimental descriptors model

Uses experimental descriptors with an option to include MACCSKeys.

#### Training on a pre-split dataset (example with MACCSKeys)

```bash
bash Experiments/solubility/experimental/train_main.sh
```

#### Investigation of model performance with new solvents (example with MACCSKeys)

```bash
bash Experiments/solubility/experimental/new_solvents_trains_maccskeys.sh
```

### Output layout

```
Models/solubility/{theoretical,experimental}_descriptors/<experiment_name>/
    fold_<n>/best_model, losses.json, metrics.json
    model_structure.json, model_config.torch
    metrics.json (aggregated over folds), test_pred, test_true (values for simple visualization)
```

Test metrics are computed for the ensemble of per-fold models, whose prediction is
the mean across folds.