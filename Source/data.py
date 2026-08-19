import random
import logging
import numpy as np
from scipy.spatial.distance import rogerstanimoto
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.cluster import KMeans, DBSCAN
from sklearn.utils.extmath import row_norms
from collections import defaultdict, Counter
import sklearn
from torch_geometric.loader import DataLoader
from rdkit import Chem
from typing import Optional
from pathlib import Path
from Source.experiment_utils import (
    _get_e_value,
    _composite_labels,
    _merge_rare_labels,
    _get_mw,
    _only_exp,
    _save_plots,
    _save_statistics,
)
 
logger = logging.getLogger(__name__)

class TaniKMeans(sklearn.cluster.KMeans):
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


def train_test_valid_split(dataset, n_folds, test_ratio=0.2, batch_size=64, seed=17):
    """
    Makes KFold cross-validation

    Parameters
    ----------
    dataset : Dataset
    n_folds : int, optional
        Number of folds in cross-validatoin
    test_ratio : float from 0.0 to 1.0, optional
        Percentage of test data in dataset
    batch_size : int, optional

    Returns
    -------
    folds : list
        List of cross-validation folds in format (train_loader, valid_loader)
    test_loader : DataLoader
        Test DataLoader, which does not participate in cross-validation
    """
    dataset_size = len(dataset)
    ids = range(dataset_size)
    train_val_ids, test_ids = train_test_split(ids, test_size=test_ratio, random_state=seed) if test_ratio > 0 else (
        ids, [])
    test_loader = DataLoader([val for i, val in enumerate(dataset) if i in test_ids], batch_size=batch_size)

    if n_folds == 1:
        train_ids, val_ids = train_test_split(train_val_ids, test_size=test_ratio, random_state=seed)
        train_loader = DataLoader([val for i, val in enumerate(dataset) if i in train_ids], batch_size=batch_size)
        val_loader = DataLoader([val for i, val in enumerate(dataset) if i in val_ids], batch_size=batch_size)
        return ((train_loader, val_loader),), test_loader

    folds = []
    kf_split = KFold(n_splits=n_folds)
    for train_index, valid_index in kf_split.split(train_val_ids):
        train_loader = DataLoader([val for i, val in enumerate(dataset) if i in train_index], batch_size=batch_size)
        valid_loader = DataLoader([val for i, val in enumerate(dataset) if i in valid_index], batch_size=batch_size)
        folds += [(train_loader, valid_loader)]
    return folds, test_loader

def solvent_stratified_split(
    dataset,  
    batch_size,
    n_folds=5, 
    shuffle_every_epoch=True, 
    seed=42, 
    test_size=0.1
):
    """
    Performs a split into folds and a test set stratified by solvent and redox_type;
    solvents with fewer than 10 molecules go entirely to the training part.

    Args:
        dataset (list): list of torch_geometric.data.Data, each must have solvent_smiles and redox_type
        n_folds (int): number of folds
        batch_size (int): batch size
        shuffle_every_epoch (bool): shuffle batches at every epoch
        seed (int): random seed
        test_size (float): fraction of the held-out set

    Returns:
        folds (list): list of (train_loader, val_loader)
        test_loader: held-out DataLoader
    """
    for d in dataset:
        if not hasattr(d, "solvent_smiles"):
            raise ValueError("Every graph must contain the solvent_smiles field.")
        if not hasattr(d, "redox_type"):
            raise ValueError("Every graph must contain the redox_type field.")

    random.seed(seed)
    np.random.seed(seed)

    solvent_groups = defaultdict(list)
    for idx, d in enumerate(dataset):
        solvent_groups[d.solvent_smiles].append(idx)

    major_idxs, minor_idxs = [], []
    for solvent, idxs in solvent_groups.items():
        if len(idxs) >= 10:
            major_idxs.extend(idxs)
        else:
            minor_idxs.extend(idxs)

    major_composite_labels = []
    for idx in major_idxs:
        d = dataset[idx]
        # Build a unique label for each solvent-redox_type combination
        composite_label = f"{d.solvent_smiles}_{d.redox_type}"
        major_composite_labels.append(composite_label)
    
    train_val_major, test_major = train_test_split(
        major_idxs, 
        test_size=test_size, 
        stratify=major_composite_labels, 
        random_state=seed
    )

    test_loader = DataLoader([dataset[i] for i in test_major], 
                             batch_size=batch_size, 
                             shuffle=False)

    # Add minor molecules to the training part only
    train_val_all = train_val_major + minor_idxs
    
    train_val_composite_labels = []
    for idx in train_val_major:
        d = dataset[idx]
        composite_label = f"{d.solvent_smiles}_{d.redox_type}"
        train_val_composite_labels.append(composite_label)
    
    minor_composite_labels = []
    for idx in minor_idxs:
        d = dataset[idx]
        minor_composite_labels.append(f"minor_{d.redox_type}")

    all_composite_labels = train_val_composite_labels + minor_composite_labels

    redox_counts = Counter([d.redox_type for d in dataset])
    redox_total = len(dataset)
    
    print(f"Original redox_type distribution:")
    for redox_type, count in redox_counts.items():
        percentage = count / redox_total * 100
        print(f"  {redox_type}: {count} ({percentage:.1f}%)")
    
    test_redox_counts = Counter([dataset[i].redox_type for i in test_major])
    test_total = len(test_major)
    
    print(f"\nredox_type distribution in the test set:")
    for redox_type, count in test_redox_counts.items():
        percentage = count / test_total * 100 if test_total > 0 else 0
        print(f"  {redox_type}: {count} ({percentage:.1f}%)")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    
    for fold_idx, (train_i, val_i) in enumerate(skf.split(train_val_all, all_composite_labels)):
        train_data = [dataset[train_val_all[i]] for i in train_i]
        val_data = [dataset[train_val_all[i]] for i in val_i if all_composite_labels[i] != "minor"]
        train_redox_counts = Counter([d.redox_type for d in train_data])
        val_redox_counts = Counter([d.redox_type for d in val_data])
        
        train_total = len(train_data)
        val_total = len(val_data)
        
        print(f"\nFold {fold_idx + 1}:")
        print(f"  Train: ", end="")
        for redox_type, count in train_redox_counts.items():
            percentage = count / train_total * 100 if train_total > 0 else 0
            print(f"{redox_type}={count}({percentage:.1f}%) ", end="")
        
        print(f"\n  Val: ", end="")
        for redox_type, count in val_redox_counts.items():
            percentage = count / val_total * 100 if val_total > 0 else 0
            print(f"{redox_type}={count}({percentage:.1f}%) ", end="")
        
        train_loader = DataLoader(train_data, 
                                 batch_size=batch_size, 
                                 shuffle=shuffle_every_epoch)
        val_loader = DataLoader(val_data, 
                               batch_size=batch_size, 
                               shuffle=False) 
        
        folds.append((train_loader, val_loader))
    
    print(f"\nCreated {len(folds)} folds for cross-validation")
    print(f"Test set: {len(test_major)} samples")
    
    return folds, test_loader


def cv_train_val_test_split(
    full_dataset: list,
    cv_folds: int,
    batch_size: int,
    test_size: float = 0.20,
    min_train_samples: int = 5,
    seed: int = 42,
    n_bins_mw: int = 4,
    n_bins_e: int = 4,
    stratify_folds: bool = True,
    output_dir: Optional[str | Path] = None,
) -> tuple[list[tuple], DataLoader]:
    """
    Split a multi-fidelity dataset into CV folds + a fixed test set,
    with optional stratification by molecular weight and redox potential,
    statistics export, and diagnostic plots.

    Calc data goes entirely into train/val, except any calc samples whose
    molecule (identified by SMILES) appears in the exp test set — those are
    removed to prevent leakage through the low-fidelity branch.
    Exp data is split per-solvent: solvents with too few samples are kept
    entirely in train/val to guarantee every test solvent is seen in training.

    Within each solvent bucket the exp samples are sorted by E-value before
    the train/test cut, and test indices are chosen with a fixed step
    (systematic sampling). This ensures the test set covers the full
    potential range rather than clustering at one end.

    When stratify_folds is True (default), the K-Fold uses StratifiedKFold
    with composite labels encoding:
        solvent | MW-quantile bin | E-value-quantile bin | redox_type

    Parameters
    ----------
    full_dataset :
        List of featurized data objects. Each object must have:
          - .data_type        str   "calc" | "exp"
          - .solvent_smiles   str
          - .y                dict  {"E": Tensor} or Tensor
          - .smiles           str   (optional, needed for MW)
          - .redox_type       str   (optional, used in stratification label)
    cv_folds :
        Number of K-Fold splits.
    batch_size :
        DataLoader batch size.
    test_size :
        Target fraction of exp data per solvent to put in test.
    min_train_samples :
        Minimum exp samples a solvent must keep in train/val.
        Solvents with < (min_train_samples + 2) total are excluded from test.
    seed :
        Random seed.
    n_bins_mw :
        Number of equal-frequency bins for MW stratification (recommended 3–6).
    n_bins_e :
        Number of equal-frequency bins for E-value stratification (recommended 3–6).
    stratify_folds :
        True  — StratifiedKFold with composite MW+E+solvent+redox labels.
        False — plain KFold.
    output_dir :
        If provided, write statistics CSVs and diagnostic plots here.

    Returns
    -------
    folds : list of (train_loader, val_loader)
    test_loader : DataLoader over the fixed exp test set
    """
    calc_dataset = [dp for dp in full_dataset if dp.data_type == "calc"]
    exp_dataset  = [dp for dp in full_dataset if dp.data_type == "exp"]

    logger.info(
        "Dataset: %d total | %d calc | %d exp",
        len(full_dataset), len(calc_dataset), len(exp_dataset),
    )

    # ------------------------------------------------------------------
    # Step 1 — Per-solvent exp split (systematic over sorted E-values)
    # ------------------------------------------------------------------
    exp_by_solvent: dict[str, list] = defaultdict(list)
    for dp in exp_dataset:
        exp_by_solvent[dp.solvent_smiles].append(dp)

    exp_trainval: list = []
    exp_test:     list = []
    excluded:     list = []
    min_required = min_train_samples + 2

    for solvent, samples in sorted(exp_by_solvent.items()):
        n = len(samples)
        samples_sorted = sorted(samples, key=_get_e_value)

        if n < min_required:
            exp_trainval.extend(samples_sorted)
            excluded.append(solvent)
            logger.info("Solvent %s: %d → all to train/val (too few)", solvent, n)
            continue

        n_test     = max(1, int(n * test_size))
        n_trainval = n - n_test

        if n_trainval < min_train_samples:
            n_trainval = min_train_samples
            n_test     = n - n_trainval
            if n_test < 1:
                exp_trainval.extend(samples_sorted)
                excluded.append(solvent)
                logger.info(
                    "Solvent %s: %d → all to train/val (can't guarantee min=%d)",
                    solvent, n, min_train_samples,
                )
                continue

        step = n / n_test
        test_indices = {int(i * step) for i in range(n_test)}
        for i, dp in enumerate(samples_sorted):
            if i in test_indices:
                exp_test.append(dp)
            else:
                exp_trainval.append(dp)

        logger.info(
            "Solvent %s: %d → %d train/val / %d test (E-sorted systematic)",
            solvent, n, n_trainval, n_test,
        )

    # ------------------------------------------------------------------
    # Step 2 — Guarantee: every test solvent must be in train/val
    # ------------------------------------------------------------------
    trainval_solvents = {dp.solvent_smiles for dp in exp_trainval}
    test_solvents     = {dp.solvent_smiles for dp in exp_test}
    missing = test_solvents - trainval_solvents
    if missing:
        raise RuntimeError(
            f"Guarantee violated: {len(missing)} test solvents not in train/val: {missing}"
        )

    logger.info(
        "Exp split: %d train/val | %d test | %d excluded | coverage %d/%d",
        len(exp_trainval), len(exp_test), len(excluded),
        len(test_solvents & trainval_solvents), len(test_solvents),
    )

    # ------------------------------------------------------------------
    # Step 2.5 — Remove all data (exp + calc) for test molecules from train/val.
    # A molecule is identified by its canonical SMILES (.smiles attribute).
    # This prevents any information about test molecules leaking through
    # the low-fidelity (calc) branch of the multifidelity dataset.
    # ------------------------------------------------------------------
    test_smiles: set[str] = {dp.smiles for dp in exp_test if hasattr(dp, "smiles")}

    if test_smiles:
        n_exp_before  = len(exp_trainval)
        n_calc_before = len(calc_dataset)

        exp_trainval = [dp for dp in exp_trainval if not (hasattr(dp, "smiles") and dp.smiles in test_smiles)]
        calc_dataset = [dp for dp in calc_dataset if not (hasattr(dp, "smiles") and dp.smiles in test_smiles)]

        n_exp_removed  = n_exp_before  - len(exp_trainval)
        n_calc_removed = n_calc_before - len(calc_dataset)

        logger.info(
            "Molecule-level leakage removal: %d test molecules | "
            "removed %d exp + %d calc from train/val",
            len(test_smiles), n_exp_removed, n_calc_removed,
        )
    else:
        logger.warning(
            "No .smiles attribute found on test data points — "
            "molecule-level leakage removal skipped."
        )

    # ------------------------------------------------------------------
    # Step 3 — K-Fold over calc + exp_trainval
    # ------------------------------------------------------------------
    train_val_dataset = calc_dataset + exp_trainval
    indices = np.arange(len(train_val_dataset))
    folds: list[tuple] = []

    if stratify_folds:
        logger.info(
            "Building composite labels (n_bins_mw=%d, n_bins_e=%d)…",
            n_bins_mw, n_bins_e,
        )
        labels = _composite_labels(train_val_dataset, n_bins_mw, n_bins_e)
        labels = _merge_rare_labels(labels, min_count=cv_folds)

        unique_labels, label_counts = np.unique(labels, return_counts=True)
        logger.info(
            "Composite labels: %d unique | min=%d | max=%d",
            len(unique_labels), label_counts.min(), label_counts.max(),
        )
        splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        splits   = list(splitter.split(indices, labels))
    else:
        splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        splits   = list(splitter.split(indices))

    folds_raw: list[tuple[list, list]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_fold = [train_val_dataset[i] for i in train_idx]
        val_fold   = [train_val_dataset[i] for i in val_idx]
        folds_raw.append((train_fold, val_fold))

        calc_t = sum(1 for dp in train_fold if dp.data_type == "calc")
        exp_t  = sum(1 for dp in train_fold if dp.data_type == "exp")
        exp_v  = sum(1 for dp in val_fold   if dp.data_type == "exp")
        train_e  = np.array([_get_e_value(dp) for dp in train_fold])
        val_e    = np.array([_get_e_value(dp)  for dp in val_fold])
        train_mw = np.array([_get_mw(dp)       for dp in train_fold])
        val_mw   = np.array([_get_mw(dp)       for dp in val_fold])

        logger.info(
            "Fold %d — train=%d (%d calc + %d exp) | val=%d (%d exp) | "
            "E: train=[%.2f,%.2f] μ=%.2f  val=[%.2f,%.2f] μ=%.2f | "
            "MW: train=[%.0f,%.0f] μ=%.0f  val=[%.0f,%.0f] μ=%.0f",
            fold_idx + 1,
            len(train_fold), calc_t, exp_t,
            len(val_fold),   exp_v,
            train_e.min(), train_e.max(), train_e.mean(),
            val_e.min(),   val_e.max(),   val_e.mean(),
            train_mw.min(), train_mw.max(), train_mw.mean(),
            val_mw.min(),   val_mw.max(),   val_mw.mean(),
        )

        folds.append((
            DataLoader(train_fold, batch_size=batch_size, shuffle=True,  drop_last=False),
            DataLoader(val_fold,   batch_size=batch_size, shuffle=False, drop_last=False),
        ))

    # ---- Test loader ----
    test_loader = DataLoader(exp_test, batch_size=batch_size, shuffle=False, drop_last=False)

    if exp_test:
        test_e  = np.array([_get_e_value(dp) for dp in exp_test])
        test_mw = np.array([_get_mw(dp)      for dp in exp_test])
        logger.info(
            "Test set: %d exp | E=[%.2f,%.2f] μ=%.2f | MW=[%.0f,%.0f] μ=%.0f",
            len(exp_test),
            test_e.min(), test_e.max(), test_e.mean(),
            test_mw.min(), test_mw.max(), test_mw.mean(),
        )

    # ------------------------------------------------------------------
    # Step 4 — Statistics export + plots
    # ------------------------------------------------------------------
    if output_dir is not None:
        out = Path(output_dir)
        logger.info("Saving split statistics and plots to %s …", out)

        stats_all, stats_exp = _save_statistics(folds_raw, exp_test, out)

        _save_plots(
            folds_raw, exp_test,
            plots_dir=out / "plots" / "all",
            all_stats=stats_all,
            filter_fn=None,
            title_suffix="",
        )

        _save_plots(
            folds_raw, exp_test,
            plots_dir=out / "plots" / "exp_only",
            all_stats=stats_exp,
            filter_fn=_only_exp,
            title_suffix=" (exp only)",
        )

        logger.info("Statistics and plots saved to %s", out)

    return folds, test_loader