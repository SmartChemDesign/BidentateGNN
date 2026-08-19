import random

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


class ECFPMolFeaturizer:

    def __init__(self, radius=6, size=512):
        self.radius = radius
        self.size = size

    def featurize(self, molecule):
        """
        :param molecule: molecule object
        :param fptype: type, radius and size of fingerprint
        :type fptype: dict
        :return: molstring for ecfp fingerprint
        """
        arr = np.zeros((1,), dtype=int)
        DataStructs.ConvertToNumpyArray(
            AllChem.GetMorganFingerprintAsBitVect(
                molecule, self.radius, self.size, useFeatures=False
            ), arr
        )
        target_names = [p for p in molecule.GetPropNames() if p.startswith("logK")]
        if len(target_names) > 1: raise ValueError("several targets for molecule")
        target = torch.tensor([float(molecule.GetProp(target_names[0]))])
        return torch.tensor(arr).to(torch.float32), {"logK": target}


def featurize_sdf(path_to_sdf=None, molecules=None, featurizer=ECFPMolFeaturizer(), seed=42):
    """
    Extract molecules from .sdf file and featurize them

    Parameters
    ----------
    path_to_sdf : str
        path to .sdf file with data
        single molecule in .sdf file can contain properties like "logK_{metal}"
        each of these properties will be transformed into a different training sample
    featurizer : featurizer, optional
        instance of the class used for extracting features of organic molecule

    Returns
    -------
    features : list of torch_geometric.data objects
        list of graphs corresponding to individual molecules from .sdf file
    """
    if (path_to_sdf is None) and (molecules is None):
        raise ValueError("'path_to_sdf' or 'molecules' parameter should be stated, got neither")
    elif (path_to_sdf is not None) and (molecules is not None):
        raise ValueError("Only one source ('path_to_sdf' or 'molecules' parameter) should be stated, got both")

    mols = molecules or [mol for mol in Chem.SDMolSupplier(path_to_sdf) if mol is not None]
    dataset = [featurizer.featurize(m) for m in mols]
    random.Random(seed).shuffle(dataset)

    return dataset
