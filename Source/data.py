"""
Dataset splitting.

Redox potentials: the split is computed once on the full experimental + calculated
dataset by assign_splits() and stored as a record -> split table, so that every
model configuration (experimental-only, calculated-only, joint, multifidelity) is
trained and tested on exactly the same records and folds. split_by_assignments()
turns the dataset and the table into per-fold lists; calibrate_folds() fits the
linear calibration of the calculated potentials on the training records of each
fold; make_loaders() wraps the folds into DataLoaders. cv_train_val_test_split()
chains these steps and exports split statistics.

Stability constants (logK): balanced and Tanimoto-cluster K-fold splits.
"""

import copy
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold, train_test_split
from torch_geometric.loader import DataLoader

from Source.experiment_utils import (
    _composite_labels,
    _get_e_value,
    _merge_rare_labels,
    _only_exp,
    _save_plots,
    _save_statistics,
)

logger = logging.getLogger(__name__)

class TaniKMeans(KMeans):
    def __init__(
        self,
        n_clusters=8,
        *,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm="elkan",
    ):
        super().__init__(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            copy_x=copy_x,
            algorithm=algorithm
        )

    def _transform(self, X):
        return pairwise_distances(X, metric="jaccard", n_jobs=self.n_clusters)



def root_mean_squared_error(*args, **kwargs):
    return np.sqrt(mean_squared_error(*args, **kwargs))


def tanimoto_train_valid_split(datasets_smiles, datasets, n_folds, batch_size, shuffle_every_epoch, valid_size=None, seed=17):
    train = [[] for _ in range(n_folds)]
    val = [[] for _ in range(n_folds)]
    train_smiles = [[] for _ in range(n_folds)]
    val_smiles = [[] for _ in range(n_folds)]
    dataset_clusters = {i: [] for i in range(n_folds)}
    unique_smiles = list()

    for smiles in datasets_smiles:
        unique_smiles.extend(smiles)
    unique_smiles = np.unique(unique_smiles)
    unique_fingerprints = [Chem.RDKFingerprint(Chem.MolFromSmiles(i)) for i in unique_smiles]
    unique_fingerprints = np.array([np.frombuffer(i.ToBitString().encode(), 'u1') - ord('0') for i in unique_fingerprints])
    kmeans = TaniKMeans(n_clusters=n_folds).fit(unique_fingerprints)
    smiles_labels = {smiles: label for smiles, label in zip(unique_smiles, kmeans.labels_)}

    for dataset, smiles_list in zip(datasets, datasets_smiles):
        mol_ids = list(range(len(dataset)))

        for fold_ind in range(n_folds):
            train[fold_ind] += [val for val, smiles in zip(dataset, smiles_list) if smiles_labels[smiles] != fold_ind]
            val[fold_ind] += [val for val, smiles in zip(dataset, smiles_list) if smiles_labels[smiles] == fold_ind]
            
            train_smiles[fold_ind] += [smiles for smiles in smiles_list if smiles_labels[smiles] != fold_ind]
            val_smiles[fold_ind] += [smiles for smiles in smiles_list if smiles_labels[smiles] == fold_ind]

    for fold_ind in range(n_folds):
        random.Random(seed).shuffle(train[fold_ind])
        random.Random(seed).shuffle(val[fold_ind])

        random.Random(seed).shuffle(train_smiles[fold_ind])
        random.Random(seed).shuffle(val_smiles[fold_ind])


    train_loaders = [DataLoader(train[fold_ind], batch_size=batch_size, shuffle=shuffle_every_epoch)
                     for fold_ind in range(n_folds)]
    valid_loaders = [DataLoader(val[fold_ind], batch_size=batch_size, shuffle=shuffle_every_epoch)
                     for fold_ind in range(n_folds)]
    return list(zip(train_loaders, valid_loaders)), [train_smiles, val_smiles]


def balanced_train_valid_split(datasets, n_folds, batch_size, shuffle_every_epoch, valid_size=None, seed=17):
    train = [[] for _ in range(n_folds)]
    val = [[] for _ in range(n_folds)]
    for dataset in datasets:
        mol_ids = list(range(len(dataset)))
        if n_folds == 1:
            if len(mol_ids) < 2:
                train[0] += [val for i, val in enumerate(dataset) if i in mol_ids]
            else:
                train_index, valid_index = train_test_split(mol_ids, test_size=valid_size, random_state=seed,
                                                            shuffle=False)
                train[0] += [val for i, val in enumerate(dataset) if i in train_index]
                val[0] += [val for i, val in enumerate(dataset) if i in valid_index]

        else:
            if len(mol_ids) < n_folds:
                for fold_ind in range(n_folds):
                    train[fold_ind] += [val for i, val in enumerate(dataset) if i in mol_ids]
            else:
                for fold_ind, (train_index, valid_index) in enumerate(KFold(n_splits=n_folds).split(mol_ids)):
                    train[fold_ind] += [val for i, val in enumerate(dataset) if i in train_index]
                    val[fold_ind] += [val for i, val in enumerate(dataset) if i in valid_index]

        for fold_ind in range(n_folds):
            random.Random(seed).shuffle(train[fold_ind])
            random.Random(seed).shuffle(val[fold_ind])

    train_loaders = [DataLoader(train[fold_ind], batch_size=batch_size, shuffle=shuffle_every_epoch)
                     for fold_ind in range(n_folds)]
    valid_loaders = [DataLoader(val[fold_ind], batch_size=batch_size, shuffle=shuffle_every_epoch)
                     for fold_ind in range(n_folds)]
    return list(zip(train_loaders, valid_loaders))


# ---------------------------------------------------------------------------
# Redox potentials: split assignment shared by all model configurations
# ---------------------------------------------------------------------------

TEST_SPLIT = "test"
EXCLUDED_SPLIT = "excluded"
RECORD_FIELDS = ("smiles", "solvent_smiles", "redox_type", "data_type")


def record_id(dp):
    """Unique key of a record: molecule | solvent | redox type | data type."""
    return "|".join(str(getattr(dp, field)) for field in RECORD_FIELDS)


def fold_split(fold_idx):
    """Name of the split whose validation part is fold number fold_idx (1-based)."""
    return f"fold_{fold_idx}"


def assign_splits(
    full_dataset,
    cv_folds,
    test_size=0.20,
    min_train_samples=5,
    seed=42,
    n_bins_mw=4,
    n_bins_e=4,
    stratify_folds=True,
):
    """
    Assign every record of the full dataset to the test set, to a validation fold
    or to the excluded set.

    1. Experimental records are split per solvent: within each solvent the records
       are sorted by E and every k-th one goes to the test set (systematic sampling),
       so the test set covers the whole potential range. Solvents with fewer than
       min_train_samples + 2 records stay entirely in train/val, so every test
       solvent is seen in training.
    2. All records (experimental and calculated) of the test molecules are excluded
       from train/val, so no information about test molecules leaks through the
       calculated branch.
    3. The remaining records are split into cv_folds folds, stratified by
       solvent | MW bin | E bin | redox type when stratify_folds is True.

    Args:
        full_dataset (list): featurized records with smiles, solvent_smiles,
            redox_type, data_type ("exp" | "calc") and y.
        cv_folds (int): number of folds.
        test_size (float): fraction of experimental records per solvent put in test.
        min_train_samples (int): minimum experimental records a solvent keeps in train/val.
        seed (int): random seed of the fold split.
        n_bins_mw (int): number of quantile bins of molecular weight for stratification.
        n_bins_e (int): number of quantile bins of E for stratification.
        stratify_folds (bool): stratified K-fold instead of plain K-fold.

    Returns:
        dict: {record_id: "test" | "fold_<k>" | "excluded"}, where "fold_<k>" means
        the record is validation data of fold k and training data of the other folds.
    """
    calc_dataset = [dp for dp in full_dataset if dp.data_type == "calc"]
    exp_dataset = [dp for dp in full_dataset if dp.data_type == "exp"]
    logger.info("Dataset: %d records, %d calc, %d exp", len(full_dataset), len(calc_dataset), len(exp_dataset))

    # Step 1: per-solvent systematic test selection over E-sorted experimental records
    exp_by_solvent = defaultdict(list)
    for dp in exp_dataset:
        exp_by_solvent[dp.solvent_smiles].append(dp)

    exp_trainval = []
    exp_test = []
    for solvent, samples in sorted(exp_by_solvent.items()):
        n = len(samples)
        # Ties in E are broken by SMILES so that the split does not depend on the record order
        samples_sorted = sorted(samples, key=lambda dp: (_get_e_value(dp), dp.smiles))
        if n < min_train_samples + 2:
            exp_trainval.extend(samples_sorted)
            logger.info("Solvent %s: %d records, all to train/val", solvent, n)
        else:
            n_test = min(max(1, int(n * test_size)), n - min_train_samples)
            step = n / n_test
            test_indices = {int(i * step) for i in range(n_test)}
            for i, dp in enumerate(samples_sorted):
                if i in test_indices:
                    exp_test.append(dp)
                else:
                    exp_trainval.append(dp)
            logger.info("Solvent %s: %d records, %d train/val, %d test", solvent, n, n - n_test, n_test)

    # Step 2: exclude every record of the test molecules from train/val
    test_smiles = {dp.smiles for dp in exp_test}
    train_val_dataset = [dp for dp in calc_dataset + exp_trainval if dp.smiles not in test_smiles]
    excluded = [dp for dp in calc_dataset + exp_trainval if dp.smiles in test_smiles]
    logger.info(
        "Test set: %d records of %d molecules; %d train/val records of these molecules excluded",
        len(exp_test), len(test_smiles), len(excluded),
    )

    # Step 3: K-fold over the remaining records
    assignments = {record_id(dp): TEST_SPLIT for dp in exp_test}
    assignments.update({record_id(dp): EXCLUDED_SPLIT for dp in excluded})
    assignments.update(assign_folds(train_val_dataset, cv_folds, seed, n_bins_mw, n_bins_e, stratify_folds))

    if len(assignments) != len(full_dataset):
        raise ValueError(
            f"Record ids are not unique: {len(full_dataset)} records, {len(assignments)} ids"
        )
    return assignments


def assign_folds(train_val_dataset, cv_folds, seed=42, n_bins_mw=4, n_bins_e=4, stratify_folds=True,
                 group_by_molecule=True):
    """
    Assign train/val records to cv_folds folds, stratified by solvent | MW bin |
    E bin | redox type when stratify_folds is True. With group_by_molecule all
    records of a molecule (every solvent, redox type and data type) go to the
    same fold, so the validation part of a fold never contains a molecule whose
    other records, in particular its calculated twins, are in the training part.
    The records are sorted by record id first, so the folds do not depend on the
    record order. Returns {record_id: "fold_<k>"}.
    """
    train_val_dataset = sorted(train_val_dataset, key=record_id)
    indices = np.arange(len(train_val_dataset))
    groups = [dp.smiles for dp in train_val_dataset] if group_by_molecule else None
    if stratify_folds:
        labels = _merge_rare_labels(_composite_labels(train_val_dataset, n_bins_mw, n_bins_e), min_count=cv_folds)
        logger.info("Stratified %d-fold split over %d composite labels%s", cv_folds, len(set(labels)),
                    ", grouped by molecule" if group_by_molecule else "")
        if group_by_molecule:
            splits = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed).split(indices, labels, groups)
        else:
            splits = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed).split(indices, labels)
    elif group_by_molecule:
        splits = GroupKFold(n_splits=cv_folds).split(indices, groups=groups)
    else:
        splits = KFold(n_splits=cv_folds, shuffle=True, random_state=seed).split(indices)

    assignments = {}
    for fold_idx, (_, val_idx) in enumerate(splits, start=1):
        for i in val_idx:
            assignments[record_id(train_val_dataset[i])] = fold_split(fold_idx)
    return assignments

def molecule_strata(exp_dataset, n_bins_e=4, min_stratum=4):
    """
    Stratum label of every molecule for the molecule-level split: primary solvent
    (the solvent with most experimental records of the molecule) | number of
    solvents (1, 2, 3 or more) | quantile bin of the mean potential within the
    redox profile (oxidation only, reduction only, both). Strata with fewer than
    min_stratum molecules are merged into coarser ones: primary solvent and number
    of solvents, then primary solvent, then one common stratum.
    """
    records = defaultdict(list)
    for dp in exp_dataset:
        records[dp.smiles].append(dp)
    smiles = sorted(records)
    primary = {s: Counter(dp.solvent_smiles for dp in records[s]).most_common(1)[0][0] for s in smiles}
    n_solvents = {s: min(len({dp.solvent_smiles for dp in records[s]}), 3) for s in smiles}
    profile = {s: "".join(sorted({dp.redox_type for dp in records[s]})) for s in smiles}
    mean_e = {s: np.mean([_get_e_value(dp) for dp in records[s]]) for s in smiles}

    e_bin = {}
    for name in set(profile.values()):
        members = [s for s in smiles if profile[s] == name]
        edges = np.quantile([mean_e[s] for s in members], np.linspace(0, 1, n_bins_e + 1))
        for s in members:
            e_bin[s] = int(np.clip(np.searchsorted(edges, mean_e[s], side="right") - 1, 0, n_bins_e - 1))

    levels = [lambda s: f"{primary[s]}|{n_solvents[s]}|{profile[s]}{e_bin[s]}",
              lambda s: f"{primary[s]}|{n_solvents[s]}", lambda s: primary[s], lambda s: "all"]
    labels = {s: levels[0](s) for s in smiles}
    for level in levels[1:]:
        counts = Counter(labels.values())
        for s in smiles:
            if counts[labels[s]] < min_stratum:
                labels[s] = level(s)
    return labels


def fill_solvent_quota(exp_dataset, test_smiles, candidates, test_size, rare_solvents, min_test_records=3):
    """
    Add test molecules for the solvents whose test records fall short of their quota
    (test_size of their experimental records, at least min_test_records so that
    per-solvent metrics can be computed): candidate molecules with a record in the solvent are
    sorted by mean potential and sampled evenly until the quota is met (all records of
    an added molecule become test data). Solvents are processed from the smallest.
    """
    by_solvent = defaultdict(list)
    for dp in exp_dataset:
        by_solvent[dp.solvent_smiles].append(dp)
    for solvent, records in sorted(by_solvent.items(), key=lambda item: len(item[1])):
        if solvent in rare_solvents:
            continue
        target = max(min_test_records, int(round(test_size * len(records))))
        missing = target - sum(dp.smiles in test_smiles for dp in records)
        pool = defaultdict(list)
        for dp in records:
            if dp.smiles in candidates and dp.smiles not in test_smiles:
                pool[dp.smiles].append(dp)
        if missing <= 0 or not pool:
            continue
        ordered = sorted(pool, key=lambda s: (np.mean([_get_e_value(dp) for dp in pool[s]]), s))
        n_add = min(len(ordered), missing)
        step = len(ordered) / n_add
        added = [ordered[int(i * step)] for i in range(n_add)]
        test_smiles.update(added)
        logger.info("Solvent %s: %d of %d test records after the stratified draw, %d molecules added to reach %d",
                    solvent, target - missing, len(records), len(added), target)
    return test_smiles


def assign_molecule_splits(full_dataset, cv_folds, test_size=0.20, min_test_records=3, min_solvent_records=10,
                           small_solvent_test_records=2, seed=42, n_bins_mw=4, n_bins_e=4, stratify_folds=True,
                           group_by_molecule=True):
    """
    Molecule-level split with a test quota per solvent and redox process.

    Every solvent gets a test quota. The solvents are processed from the smallest to
    the largest and, within a solvent, the smaller process group (oxidation or
    reduction) first. The quota of a group is test_size of its records; the two quotas
    of a solvent sum to at least min_test_records (small_solvent_test_records for
    solvents with fewer than min_solvent_records records), the excess going to the
    larger group. A quota is filled by molecules
    with a record in the group whose other records lie only in groups not processed
    yet, sorted by the potential of that record and sampled systematically, so the
    test covers the potential range of every group. When a group runs out of such
    molecules, the shortfall below min_test_records is taken from the other group of
    the solvent. A chosen molecule brings all its experimental records to the test
    set; its calculated records are excluded; everything else is split into cv_folds
    folds by assign_folds() with seed. The selection is deterministic, so every regime
    and seed shares the test set.

    Returns:
        dict: {record_id: "test" | "fold_<k>" | "excluded"}
    """
    exp_dataset = [dp for dp in full_dataset if dp.data_type == "exp"]
    by_solvent = defaultdict(list)
    groups_of = defaultdict(set)
    for dp in exp_dataset:
        by_solvent[dp.solvent_smiles].append(dp)
        groups_of[dp.smiles].add((dp.solvent_smiles, dp.redox_type))
    order = sorted(by_solvent, key=lambda s: (len(by_solvent[s]), s))

    def candidates_of(group, blocked):
        """Records of the group whose molecule is not in the test set and has no record in a blocked group, by potential."""
        return sorted((dp for dp in group if dp.smiles not in test_smiles and not (groups_of[dp.smiles] & blocked)),
                      key=lambda dp: (_get_e_value(dp), dp.smiles))

    def pick(candidates, n):
        """Molecules of n records sampled evenly over the potential-ordered candidates."""
        n = min(len(candidates), max(0, n))
        return [candidates[int(i * len(candidates) / n)].smiles for i in range(n)]

    test_smiles = set()
    closed = set()
    for solvent in order:
        records = by_solvent[solvent]
        minimum = min_test_records if len(records) >= min_solvent_records else small_solvent_test_records
        groups = defaultdict(list)
        for dp in records:
            groups[dp.redox_type].append(dp)
        quotas = {redox: int(round(test_size * len(group))) for redox, group in groups.items()}
        largest = max(groups, key=lambda r: len(groups[r]))
        quotas[largest] += max(0, minimum - sum(quotas.values()))

        for redox in sorted(groups, key=lambda r: (len(groups[r]), r)):
            group = groups[redox]
            n_test = sum(dp.smiles in test_smiles for dp in group)
            candidates = candidates_of(group, closed)
            chosen = pick(candidates, quotas[redox] - n_test)
            test_smiles.update(chosen)
            closed.add((solvent, redox))
            logger.info("Solvent %s %s: %d records, quota %d, %d already in test, %d molecules chosen of %d candidates -> %d",
                        solvent, redox, len(group), quotas[redox], n_test, len(chosen), len(candidates),
                        sum(dp.smiles in test_smiles for dp in group))

        n_test = sum(dp.smiles in test_smiles for dp in records)
        if n_test < minimum:
            own = {(solvent, redox) for redox in groups}
            chosen = pick(candidates_of(records, closed - own), minimum - n_test)
            test_smiles.update(chosen)
            logger.info("Solvent %s: %d test records below the minimum %d, %d molecules added -> %d",
                        solvent, n_test, minimum, len(chosen), sum(dp.smiles in test_smiles for dp in records))

    exp_test = [dp for dp in exp_dataset if dp.smiles in test_smiles]
    excluded = [dp for dp in full_dataset if dp.data_type == "calc" and dp.smiles in test_smiles]
    train_val = [dp for dp in full_dataset if dp.smiles not in test_smiles]
    logger.info("Molecule split: test %d of %d molecules, %d of %d experimental records (%d ox, %d red); %d calculated "
                "records excluded; %d train/val records (%d experimental)", len(test_smiles), len(groups_of),
                len(exp_test), len(exp_dataset), sum(dp.redox_type == "ox" for dp in exp_test),
                sum(dp.redox_type == "red" for dp in exp_test), len(excluded), len(train_val),
                sum(dp.data_type == "exp" for dp in train_val))

    assignments = {record_id(dp): TEST_SPLIT for dp in exp_test}
    assignments.update({record_id(dp): EXCLUDED_SPLIT for dp in excluded})
    assignments.update(assign_folds(train_val, cv_folds, seed, n_bins_mw, n_bins_e, stratify_folds, group_by_molecule))
    if len(assignments) != len(full_dataset):
        raise ValueError(f"Record ids are not unique: {len(full_dataset)} records, {len(assignments)} ids")
    return assignments

def assign_loso_splits(
    full_dataset,
    solvent_smiles,
    cv_folds,
    seed=42,
    excluded_smiles=(),
    n_bins_mw=4,
    n_bins_e=4,
    stratify_folds=True,
    group_by_molecule=True,
):
    """
    Leave-one-solvent-out assignment: every experimental record of solvent_smiles
    is test data, its calculated records are excluded, and the records of the
    other solvents are split into cv_folds folds as in assign_splits(). The
    held-out solvent is therefore never seen in training, neither through
    experimental nor through calculated records. With excluded_smiles (e.g. the
    test molecules of the main split) every record of these molecules in the other
    solvents is excluded as well, so that the test records of the held-out solvent
    split into molecules unseen in any solvent and molecules seen in other solvents.

    Returns:
        dict: {record_id: "test" | "fold_<k>" | "excluded"}
    """
    excluded_smiles = set(excluded_smiles)
    in_solvent = [dp for dp in full_dataset if dp.solvent_smiles == solvent_smiles]
    exp_test = [dp for dp in in_solvent if dp.data_type == "exp"]
    train_val_dataset = [dp for dp in full_dataset
                         if dp.solvent_smiles != solvent_smiles and dp.smiles not in excluded_smiles]
    excluded = [dp for dp in in_solvent if dp.data_type == "calc"] + [
        dp for dp in full_dataset if dp.solvent_smiles != solvent_smiles and dp.smiles in excluded_smiles]
    logger.info(
        "Held-out solvent %s: %d test records (%d of molecules excluded from training), %d records excluded, "
        "%d train/val records", solvent_smiles, len(exp_test), sum(dp.smiles in excluded_smiles for dp in exp_test),
        len(excluded), len(train_val_dataset),
    )

    assignments = {record_id(dp): TEST_SPLIT for dp in exp_test}
    assignments.update({record_id(dp): EXCLUDED_SPLIT for dp in excluded})
    assignments.update(assign_folds(train_val_dataset, cv_folds, seed, n_bins_mw, n_bins_e, stratify_folds, group_by_molecule))
    return assignments


def save_assignments(assignments, path):
    """Save the record -> split table as CSV (one column per record field plus 'split')."""
    rows = []
    for key, split in assignments.items():
        rows.append(dict(zip(RECORD_FIELDS, key.split("|"))) | {"split": split})
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info("Saved %d split assignments to %s", len(rows), path)


def load_assignments(path):
    """Load the record -> split table written by save_assignments()."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    assignments = {
        "|".join(row[field] for field in RECORD_FIELDS): row["split"] for _, row in df.iterrows()
    }
    logger.info("Loaded %d split assignments from %s", len(assignments), path)
    return assignments


def split_by_assignments(dataset, assignments):
    """
    Distribute the records of a dataset over folds and test set according to
    the assignments. The dataset may be a subset of the one the assignments were
    computed on; assignments of absent records are ignored, records without an
    assignment raise an error.

    Returns:
        folds (list): per fold a (train_records, val_records) pair.
        test (list): test records.
    """
    missing = [record_id(dp) for dp in dataset if record_id(dp) not in assignments]
    if missing:
        raise ValueError(f"{len(missing)} records have no split assignment, e.g. {missing[0]}")

    fold_names = sorted({s for s in assignments.values() if s.startswith("fold_")},
                        key=lambda s: int(s.split("_")[1]))
    test = [dp for dp in dataset if assignments[record_id(dp)] == TEST_SPLIT]
    train_val = [dp for dp in dataset if assignments[record_id(dp)] in fold_names]
    folds = []
    for name in fold_names:
        val = [dp for dp in train_val if assignments[record_id(dp)] == name]
        train = [dp for dp in train_val if assignments[record_id(dp)] != name]
        folds.append((train, val))
        logger.info(
            "%s: train %d (%d exp), val %d (%d exp)",
            name, len(train), sum(dp.data_type == "exp" for dp in train),
            len(val), sum(dp.data_type == "exp" for dp in val),
        )
    logger.info("Test: %d records", len(test))
    return folds, test


def fit_calibration(pairs, groups, min_points, exclusion=None):
    """
    Fit E_exp = a * E_calc + b per (solvent, redox_type) from (solvent, redox_type,
    E_calc, E_exp) pairs. Every group in `groups` (all groups holding calculated
    records) gets coefficients: its own fit when it has at least min_points pairs,
    the global fit of its redox type otherwise (including groups without any pair).

    With exclusion None every group is kept. Otherwise a group is excluded when
      - own fit: p >= exclusion["max_p_value"] or R^2 < exclusion["min_r2"];
      - global fit: the global fit has R^2 < exclusion["min_global_r2"], or the mean
        absolute residual of the group's own pairs under the global line exceeds
        exclusion["max_fallback_residual"], or the group has no pair and
        exclusion["allow_no_pairs"] is False.

    Returns:
        coefficients (dict): {(solvent, redox_type): (a, b)}
        excluded (set): (solvent, redox_type) groups whose calculated records are dropped
        rows (list): one dict per group for the calibration table
    """
    df = pd.DataFrame(pairs, columns=["solvent", "redox_type", "E_calc", "E_exp"])

    global_fits = {}
    for redox_type, group in df.groupby("redox_type"):
        fit = stats.linregress(group["E_calc"], group["E_exp"])
        global_fits[redox_type] = (fit.slope, fit.intercept, fit.rvalue ** 2)

    coefficients = {}
    excluded = set()
    rows = []
    for solvent, redox_type in sorted(groups):
        group = df[(df["solvent"] == solvent) & (df["redox_type"] == redox_type)]
        residual = np.nan
        is_excluded = False
        if len(group) >= min_points:
            fit = stats.linregress(group["E_calc"], group["E_exp"])
            a, b, r2, p_value, source = fit.slope, fit.intercept, fit.rvalue ** 2, fit.pvalue, "per_solvent"
            if exclusion is not None:
                is_excluded = p_value >= exclusion["max_p_value"] or r2 < exclusion["min_r2"]
        else:
            a, b, global_r2 = global_fits[redox_type]
            r2, p_value, source = np.nan, np.nan, "global_fallback"
            if len(group):
                residual = float(np.mean(np.abs(group["E_exp"] - (a * group["E_calc"] + b))))
            if exclusion is not None:
                is_excluded = (global_r2 < exclusion["min_global_r2"]
                               or (len(group) == 0 and not exclusion["allow_no_pairs"])
                               or (len(group) > 0 and residual > exclusion["max_fallback_residual"]))
        coefficients[(solvent, redox_type)] = (a, b)
        if is_excluded:
            excluded.add((solvent, redox_type))
        rows.append({"solvent": solvent, "redox_type": redox_type, "a": a, "b": b,
                     "r2": r2, "p_value": p_value, "n_points": len(group),
                     "source": source, "fallback_residual": residual, "excluded": is_excluded})
    return coefficients, excluded, rows


def calibrate_folds(folds, min_points=5, exclusion=None, exclude_across_folds=True, output_dir=None):
    """
    Calibrate the calculated (LF) potentials of every fold on that fold's
    training records only.

    The dataset holds a calculated record for every molecule, solvent and redox
    type, including the combinations that also have an experimental record. These
    calculated twins provide the (E_calc, E_exp) pairs: for fold k the pairs are
    the experimental training records of the fold joined with their twins by
    (smiles, solvent_smiles, redox_type). The fitted coefficients are applied to
    the remaining calculated records of the train and validation parts; records of
    excluded (solvent, redox_type) groups and all twins of experimental records are
    dropped, so the network never sees two targets for one input. Experimental and
    test records are never modified.

    Exclusion is decided per fold by fit_calibration(); with exclude_across_folds
    a group excluded in any fold is excluded in every fold, so all fold models are
    trained on the same set of groups. The coefficients stay fold-specific.

    Calculated records are copied before their target is replaced, because one
    record object is shared between folds.

    With output_dir, writes per fold calibration_fold_<k>.csv (coefficients per
    group, the fold's own exclusion decision, the final one, and the fate of the
    group's calculated train/val records: n_calc, n_dropped_twin,
    n_dropped_excluded, n_kept) and calibrated_fold_<k>.csv (raw and calibrated
    potential of every calculated record kept in the fold).

    Returns:
        list: calibrated (train_records, val_records) per fold.
    """
    def key(dp):
        return (dp.smiles, dp.solvent_smiles, dp.redox_type)

    # Pass 1: fit the calibration of every fold on its training records
    fits = []
    for fold_idx, (train, val) in enumerate(folds, start=1):
        calc_by_key = {key(dp): _get_e_value(dp) for dp in train + val if dp.data_type == "calc"}
        pairs = [(dp.solvent_smiles, dp.redox_type, calc_by_key[key(dp)], _get_e_value(dp))
                 for dp in train if dp.data_type == "exp" and key(dp) in calc_by_key]
        if not pairs:
            raise ValueError(f"Fold {fold_idx}: no experimental training records with a calculated twin")
        calc_groups = {(dp.solvent_smiles, dp.redox_type) for dp in train + val if dp.data_type == "calc"}
        fits.append((pairs, calc_groups, *fit_calibration(pairs, calc_groups, min_points, exclusion)))

    excluded_union = set().union(*(excluded for _, _, _, excluded, _ in fits))
    if exclude_across_folds and excluded_union:
        logger.info("Groups excluded in at least one fold are excluded in all folds: %s",
                    ", ".join(f"{solvent} {redox_type}" for solvent, redox_type in sorted(excluded_union)))

    # Pass 2: apply the coefficients and drop excluded groups and twins
    calibrated_folds = []
    for fold_idx, ((train, val), (pairs, calc_groups, coefficients, excluded_in_fold, rows)) in enumerate(zip(folds, fits), start=1):
        excluded = excluded_union if exclude_across_folds else excluded_in_fold
        exp_keys = {key(dp) for dp in train + val if dp.data_type == "exp"}
        calibrated_rows = []
        counts = {group: defaultdict(int) for group in calc_groups}

        def calibrate(records, part):
            result = []
            for dp in records:
                group = (dp.solvent_smiles, dp.redox_type)
                if dp.data_type != "calc":
                    result.append(dp)
                elif key(dp) in exp_keys:
                    counts[group]["n_dropped_twin"] += 1
                elif group in excluded:
                    counts[group]["n_dropped_excluded"] += 1
                else:
                    counts[group]["n_kept"] += 1
                    a, b = coefficients[group]
                    e_raw = _get_e_value(dp)
                    dp = copy.copy(dp)
                    dp.y = {"E": torch.tensor([[a * e_raw + b]], dtype=torch.float32)}
                    result.append(dp)
                    calibrated_rows.append({
                        "smiles": dp.smiles, "solvent_smiles": dp.solvent_smiles,
                        "redox_type": dp.redox_type, "part": part,
                        "E_raw": e_raw, "E_calibrated": a * e_raw + b,
                    })
            return result

        train_cal, val_cal = calibrate(train, "train"), calibrate(val, "val")
        for row in rows:
            group = (row["solvent"], row["redox_type"])
            row["excluded_in_fold"] = row.pop("excluded")
            row["excluded"] = group in excluded
            row.update({name: counts[group][name] for name in ("n_dropped_twin", "n_dropped_excluded", "n_kept")})
            row["n_calc"] = sum(counts[group].values())
        logger.info(
            "fold_%d calibration: %d pairs, %d groups, %d excluded (%s); calc records train %d -> %d, val %d -> %d",
            fold_idx, len(pairs), len(coefficients), len(excluded),
            ", ".join(f"{solvent} {redox_type}" for solvent, redox_type in sorted(excluded)) or "none",
            sum(dp.data_type == "calc" for dp in train), sum(dp.data_type == "calc" for dp in train_cal),
            sum(dp.data_type == "calc" for dp in val), sum(dp.data_type == "calc" for dp in val_cal),
        )
        calibrated_folds.append((train_cal, val_cal))

        if output_dir is not None:
            pd.DataFrame(rows).sort_values(["redox_type", "solvent"]).to_csv(
                Path(output_dir) / f"calibration_fold_{fold_idx}.csv", index=False, float_format="%.6f")
            pd.DataFrame(calibrated_rows).to_csv(
                Path(output_dir) / f"calibrated_fold_{fold_idx}.csv", index=False, float_format="%.6f")
    return calibrated_folds


def make_loaders(folds, test, batch_size):
    """Wrap per-fold record lists and the test records into DataLoaders."""
    fold_loaders = [
        (DataLoader(train, batch_size=batch_size, shuffle=True),
         DataLoader(val, batch_size=batch_size, shuffle=False))
        for train, val in folds
    ]
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return fold_loaders, test_loader


def cv_train_val_test_split(
    full_dataset,
    cv_folds,
    batch_size,
    test_size=0.20,
    min_train_samples=5,
    seed=42,
    n_bins_mw=4,
    n_bins_e=4,
    stratify_folds=True,
    assignments=None,
    calibration=None,
    train_data_types=None,
    output_dir: Optional[str | Path] = None,
):
    """
    Split a dataset into CV folds and a test set and build the DataLoaders.

    When assignments is given, the split is taken from it (see assign_splits and
    load_assignments); otherwise it is computed on full_dataset. calibration is a
    dict of calibrate_folds() parameters; when given, the calculated potentials
    of every fold are calibrated on that fold's training records before the
    data-type filter. train_data_types restricts the train and validation folds
    to these data types ("exp", "calc"); the test set is not affected. When
    output_dir is given, the assignments, calibration tables, split statistics
    and diagnostic plots are written there.

    Returns:
        folds (list): per fold a (train_loader, val_loader) pair.
        test_loader (DataLoader): the test set.
    """
    if assignments is None:
        assignments = assign_splits(
            full_dataset, cv_folds, test_size, min_train_samples, seed,
            n_bins_mw, n_bins_e, stratify_folds,
        )
    folds, test = split_by_assignments(full_dataset, assignments)
    out = None
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_assignments(assignments, out / "split_assignments.csv")

    if calibration is not None:
        folds = calibrate_folds(folds, output_dir=out, **calibration)
    if train_data_types is not None:
        if isinstance(train_data_types, str):
            raise TypeError(f"train_data_types must be a tuple such as ('calc',), got {train_data_types!r}")
        folds = [([dp for dp in train if dp.data_type in train_data_types],
                  [dp for dp in val if dp.data_type in train_data_types])
                 for train, val in folds]
        for fold_idx, (train, val) in enumerate(folds, start=1):
            logger.info("fold_%d after keeping %s: train %d (%d exp, %d calc), val %d (%d exp, %d calc)",
                        fold_idx, "+".join(train_data_types),
                        len(train), sum(dp.data_type == "exp" for dp in train), sum(dp.data_type == "calc" for dp in train),
                        len(val), sum(dp.data_type == "exp" for dp in val), sum(dp.data_type == "calc" for dp in val))

    if out is not None:
        stats_all, stats_exp = _save_statistics(folds, test, out)
        _save_plots(folds, test, plots_dir=out / "plots" / "all", all_stats=stats_all)
        _save_plots(folds, test, plots_dir=out / "plots" / "exp_only", all_stats=stats_exp,
                    filter_fn=_only_exp, title_suffix=" (exp only)")

    return make_loaders(folds, test, batch_size)
