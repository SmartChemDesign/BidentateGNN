import random

import numpy as np
from scipy.spatial.distance import rogerstanimoto
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import KFold, train_test_split
from sklearn.cluster import KMeans, DBSCAN
from sklearn.utils.extmath import row_norms
import sklearn
from torch_geometric.loader import DataLoader
from rdkit import Chem


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
