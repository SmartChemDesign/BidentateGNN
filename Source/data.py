import random

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from torch_geometric.loader import DataLoader


def root_mean_squared_error(*args, **kwargs):
    return np.sqrt(mean_squared_error(*args, **kwargs))


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
