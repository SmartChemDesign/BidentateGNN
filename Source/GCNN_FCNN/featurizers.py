import sys
import copy
import json
import random
import warnings
import pandas as pd
from typing import Union, Optional

import dgl
import torch
from dgllife.utils import mol_to_bigraph
from rdkit import Chem
from rdkit.Chem import Mol
from torch_geometric.utils import from_networkx
from sklearn.preprocessing import StandardScaler
from Source.GCNN_FCNN.old_featurizer import ConvMolFeaturizer
from Source.config import ROOT_DIR

def load_solvent_info(base_path, solvent_mode):
    solvent_info = {"mode": solvent_mode}
    
    if solvent_mode == "onehot":
        vocab_path = f"{base_path}_solvent_vocab.csv"
        df = pd.read_csv(vocab_path)
        solvent_vocab = dict(zip(df["solvent_smiles"], df["index"]))
        solvent_info["vocab"] = solvent_vocab
        solvent_info["n_features"] = len(solvent_vocab) + 1
    
    elif solvent_mode == "descriptors":
        scaler_path = f"{base_path}_solvent_scaler.pkl"
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        solvent_info["scaler"] = scaler
        solvent_info["n_features"] = 8
    
    return solvent_info


def convert_to_float32(data_list):
    for data in data_list:
        if hasattr(data, 'x') and torch.is_tensor(data.x):
            data.x = data.x.float()
        if hasattr(data, 'y'):
            if isinstance(data.y, dict):
                for key, value in data.y.items():
                    if torch.is_tensor(value):
                        data.y[key] = value.float()
            elif torch.is_tensor(data.y):
                data.y = data.y.float()
        if hasattr(data, 'solvent_x') and torch.is_tensor(data.solvent_x):
            data.solvent_x = data.solvent_x.float()
        if hasattr(data, 'edge_attr') and torch.is_tensor(data.edge_attr):
            data.edge_attr = data.edge_attr.float()
        if hasattr(data, 'edge_index') and torch.is_tensor(data.edge_index):
            if data.edge_index.dtype == torch.float64:
                data.edge_index = data.edge_index.float()
    
    return data_list

class DGLFeaturizer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def featurize(self, mol, require_node_features=True, require_edge_features=True):
        dgl_graph = mol_to_bigraph(mol, **self.kwargs)
        networkx_graph = dgl.to_networkx(dgl_graph)
        graph = from_networkx(networkx_graph)
        if 'h' not in dgl_graph.ndata:
            if require_node_features:
                warnings.warn(f"can't featurize {Chem.MolToSmiles(mol)}: 'h' not in graph.ndata. Skipping.")
                return None
            else:
                warnings.warn(f"No node_features in {Chem.MolToSmiles(mol)}")
                dgl_graph.ndata['h'] = None
        if 'e' not in dgl_graph.edata:
            if require_edge_features:
                warnings.warn(f"can't featurize {Chem.MolToSmiles(mol)}: 'e' not in graph.edata. Skipping.")
                return None
            else:
                warnings.warn(f"No edge_features in {Chem.MolToSmiles(mol)}")
                dgl_graph.edata['e'] = None
            return None
        graph.x = dgl_graph.ndata['h']
        graph.edge_attr = dgl_graph.edata['e']
        graph.id = None
        return graph


class SkipatomFeaturizer:
    """
    Class for extracting element features by skipatom_models approach

    Attributes
    ----------
    get_vector : dict
        skipatom vectors for each element

    Methods
    ----------
    _featurize(element : str)
        get skipatom features for given element
    """

    def __init__(self, vectors_filename=ROOT_DIR / "Source/GCNN_FCNN/skipatom_vectors_dim200.json"):
        with open(vectors_filename, "r") as f:
            self.get_vector = json.load(f)

    def featurize(self, element):
        """

        Parameters
        ----------
        element : str
            element to be featurized

        Returns
        -------
            features : torch.tensor
                features of an element obtained from skipatom approach, shape (1, 200)
        """
        return torch.tensor(self.get_vector[element]).unsqueeze(0)


def featurize_sdf_with_metal(path_to_sdf=None, molecules=None, mol_featurizer=ConvMolFeaturizer(),
                             metal_featurizer=SkipatomFeaturizer(),
                             seed=42):
    """
    Extract molecules from .sdf file and featurize them

    Parameters
    ----------
    path_to_sdf : str
        path to .sdf file with data
        single molecule in .sdf file can contain properties like "logK_{metal}"
        each of these properties will be transformed into a different training sample
    mol_featurizer : featurizer, optional
        instance of the class used for extracting features of organic molecule
    metal_featurizer : featurizer, optional
        instance of the class used for extracting metal features

    Returns
    -------
    features : list of torch_geometric.data objects
        list of graphs corresponding to individual molecules from .sdf file
    """
    if path_to_sdf is None and molecules is None:
        raise ValueError("'path_to_sdf' or 'molecules' parameter should be stated, got neither")
    elif path_to_sdf is not None and molecules is not None:
        raise ValueError("Only one source ('path_to_sdf' or 'molecules' parameter) should be stated, got both")
    mols = molecules or [mol for mol in Chem.SDMolSupplier(path_to_sdf) if mol is not None]
    mol_graphs = [mol_featurizer.featurize(m) for m in mols]

    all_data = []
    for mol, graph in zip(mols, mol_graphs):
        targets = [prop for prop in mol.GetPropNames() if prop.startswith("logK_")]
        for target in targets:
            new_graph = copy.deepcopy(graph)

            element_symbol = target.split("_")[-1]
            new_graph.metal_x = metal_featurizer.featurize(element_symbol)
            new_graph.y = {"logK": torch.tensor([[float(mol.GetProp(target))]])}
            all_data += [new_graph]
    random.Random(seed).shuffle(all_data)

    return all_data


def featurize_sdf_with_metal_and_conditions(path_to_sdf=None, molecules=None, mol_featurizer=ConvMolFeaturizer(),
                                            metal_featurizer=SkipatomFeaturizer(), seed=42, shuffle=True):
    """
    Extract molecules from .sdf file and featurize them

    Parameters
    ----------
    path_to_sdf : str
        path to .sdf file with data
        single molecule in .sdf file can contain properties like "logK_{metal}"
        each of these properties will be transformed into a different training sample
    mol_featurizer : featurizer, optional
        instance of the class used for extracting features of organic molecule
    metal_featurizer : featurizer, optional
        instance of the class used for extracting metal features
    data_multy_coefficients: dict
        each complex of metal Me will be used data_multy_coefficients[Me] times

    Returns
    -------
    features : list of torch_geometric.data objects
        list of graphs corresponding to individual molecules from .sdf file
    """
    if path_to_sdf is None and molecules is None:
        raise ValueError("'path_to_sdf' or 'molecules' parameter should be stated, got neither")
    elif path_to_sdf is not None and molecules is not None:
        raise ValueError("Only one source ('path_to_sdf' or 'molecules' parameter) should be stated, got both")
    mols = molecules or [mol for mol in Chem.SDMolSupplier(path_to_sdf) if mol is not None]
    smiles = [Chem.MolToSmiles(i) for i in mols]

    mol_features = [mol_featurizer.featurize(m) for m in mols]
    all_data = []
    for mol_ind in range(len(mols)):
        metals = []
        conditions = []
        logKs = []
        for target in [prop for prop in mols[mol_ind].GetPropNames() if prop.startswith("logK_")]:
            element_symbol, charge_str, temperature_str, ionic_str_str = target.split("_")[1:]
            charge = float(charge_str.split("=")[-1])
            temperature = float(temperature_str.split("=")[-1])
            ionic_str = float(ionic_str_str.split("=")[-1])
            metals += [element_symbol]
            conditions += [torch.tensor([[charge, temperature, ionic_str]])]
            logKs += [float(mols[mol_ind].GetProp(target))]
        for element_symbol, condition_values, logK in zip(metals, conditions, logKs):
            features = copy.deepcopy(mol_features[mol_ind])
            features.metal_x = torch.cat((metal_featurizer.featurize(element_symbol), condition_values), dim=-1)
            features.y = {"logK": torch.tensor([[logK]])}
            all_data += [features]

    if shuffle: 
        random.Random(seed).shuffle(all_data)
        random.Random(seed).shuffle(smiles)

    return all_data, smiles


class Complex:
    def __init__(self, mol: Union[str, Mol], metal: str,
                 valence: int, temperature: float, ionic_str: float,
                 logk: Optional[float] = None):

        self.metal = metal
        self.valence = valence if valence else 3
        self.temperature = temperature if temperature else 20
        self.ionic_str = ionic_str if ionic_str else 0.1
        self.logk = logk
        if isinstance(mol, str):
            self.mol = Chem.MolFromSmiles(mol)
        elif isinstance(mol, Mol):
            self.mol = mol
        else:
            raise ValueError(f"invalid molecule input type: {type(mol)}")

        self.mol_featurizer = ConvMolFeaturizer()
        self.metal_featurizer = SkipatomFeaturizer()

        self.graph = self.mol_featurizer.featurize(self.mol)
        conditions = torch.tensor([[self.valence, self.temperature, self.ionic_str]])
        self.graph.metal_x = torch.cat((self.metal_featurizer.featurize(self.metal), conditions), dim=-1)
        if self.logk:
            self.graph.y = torch.tensor([self.logk])


# Columns of the solvent table used as solvent descriptors, in solvent_x order:
# the full set of eight physical properties
SOLVENT_DESCRIPTOR_COLUMNS = {
    "Molecular weight": "molecular_weight",
    "Vapor pressure, mmHg(25C)": "vapor_pressure",
    "Refractive index, n25/D": "refractive_index",
    "Boiling point, C": "boiling_point",
    "Melting point, C": "melting_point",
    "Viscosity, cp(25C)": "viscosity",
    "Density, g/mL(25C)": "density",
    "Dielectric permittivity(25C)": "dielectric",
}

EXCLUDED_DESCRIPTORS = {name.strip() for name in os.environ.get("EXCLUDE_SOLVENT_DESCRIPTORS", "").split(",") if name.strip()}
SOLVENT_DESCRIPTOR_COLUMNS = {column: short for column, short in SOLVENT_DESCRIPTOR_COLUMNS.items()
                              if short not in EXCLUDED_DESCRIPTORS}

SOLVENT_DESCRIPTORS_OFF = bool(os.environ.get("SOLVENT_DESCRIPTORS_OFF"))

def load_solvent_db(path):
    """
    Load the solvent table (columns 'solvent', 'smiles' and the descriptor columns).

    Returns:
        solvent_db (dict): {canonical_smiles: [descriptor values]}
        name_to_smiles (dict): {solvent name: canonical_smiles}
    """
    df = pd.read_csv(path)
    solvent_db = {}
    name_to_smiles = {}
    for _, row in df.iterrows():
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(str(row["smiles"]).strip()))
        solvent_db[smiles] = [float(row[col]) if pd.notna(row[col]) else 0.0
                              for col in SOLVENT_DESCRIPTOR_COLUMNS]
        name_to_smiles[str(row["solvent"]).strip()] = smiles
    logger.info("Loaded %d solvents from %s", len(solvent_db), path)
    return solvent_db, name_to_smiles


def resolve_solvent_smiles(mol, redox_type, name_to_smiles):
    """
    Canonical solvent SMILES of a record: solvent_smiles_{redox_type} has priority,
    solvent_{redox_type} (name from the solvent table) is the fallback.
    Returns None when neither property resolves.
    """
    props = mol.GetPropNames()
    smiles_prop = f"solvent_smiles_{redox_type}"
    name_prop = f"solvent_{redox_type}"
    if smiles_prop in props:
        solvent_mol = Chem.MolFromSmiles(mol.GetProp(smiles_prop).strip())
        if solvent_mol is not None:
            return Chem.MolToSmiles(solvent_mol)
    if name_prop in props:
        return name_to_smiles.get(mol.GetProp(name_prop).strip())
    return None


def read_redox_record(mol, mol_idx, solvent_db, name_to_smiles):
    """
    Parse one SDF molecule into (E, solvent_smiles, redox_type, data_type).
    Returns None when the record is skipped; the reason is logged.
    """
    props = mol.GetPropNames()
    mol_smiles = Chem.MolToSmiles(mol)

    if "redox_type" not in props:
        logger.warning("Molecule %d (%s): no redox_type, skipping", mol_idx + 1, mol_smiles)
        return None
    redox_type_full = mol.GetProp("redox_type").strip().lower()
    if redox_type_full == "oxidation":
        redox_type = "ox"
    elif redox_type_full == "reduction":
        redox_type = "red"
    else:
        logger.warning("Molecule %d (%s): invalid redox_type %r, skipping",
                       mol_idx + 1, mol_smiles, redox_type_full)
        return None

    solvent_smiles = resolve_solvent_smiles(mol, redox_type, name_to_smiles)
    if solvent_smiles is None or solvent_smiles not in solvent_db:
        logger.warning("Molecule %d (%s): solvent not resolved or not in the solvent table, skipping",
                       mol_idx + 1, mol_smiles)
        return None

    calc_prop = f"E_{redox_type}_V_calc"
    exp_prop = f"E_{redox_type}_V"
    if calc_prop in props:
        E = float(mol.GetProp(calc_prop))
        data_type = "calc"
    elif exp_prop in props:
        E = float(mol.GetProp(exp_prop))
        data_type = "exp"
    else:
        logger.warning("Molecule %d (%s): no E value for %s, skipping", mol_idx + 1, mol_smiles, redox_type)
        return None

    logger.debug("Molecule %d: %s, redox_type=%s, solvent=%s, E=%.4f (%s)",
                 mol_idx + 1, mol_smiles, redox_type, solvent_smiles, E, data_type)
    return E, solvent_smiles, redox_type, data_type


def save_table(records, path):
    """Save a list of dicts as CSV or XLSX, chosen by the extension."""
    df = pd.DataFrame(records)
    if str(path).endswith(".xlsx"):
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)
    logger.info("Saved %d records to %s", len(df), path)
    return df


def featurize_sdf_with_solvent_all(
    path_to_sdf=None,
    molecules=None,
    mol_featurizer=None,
    seed=42,
    shuffle=True,
    scale_solvent=True,
    solvent_db_path=None,
    output_dataset_path=None,
    log_file=None,
    output_model_dataset_path=None,
    scaler_save_path=None):
    """
    Featurize SDF molecules with a solvent descriptor vector and redox_type.

    Each molecule carries redox_type, solvent_{redox_type} or solvent_smiles_{redox_type},
    and one of the targets: E_{redox_type}_V (experimental) or E_{redox_type}_V_calc (raw
    PM7/COSMO value, calibrated later inside the folds). A calculated target has
    priority over the experimental one. The SDF holds calculated records for every
    molecule, solvent and redox type, including combinations with an experimental record;
    these calculated twins are used for the in-fold calibration (data.calibrate_folds).

    solvent_x layout: [scaled descriptors..., redox_type_binary, data_type]
    with redox_type_binary = 1 for oxidation and data_type = 1 for experimental records.

    Args:
        path_to_sdf (str, optional): Path to SDF file.
        molecules (list, optional): List of RDKit molecules.
        mol_featurizer: Object with a featurize(mol) method.
        seed (int): Random seed for shuffling.
        shuffle (bool): Shuffle the output list.
        scale_solvent (bool): Apply StandardScaler to the descriptors (fitted on unique solvents).
        solvent_db_path (str): Path to the solvent table CSV.
        output_dataset_path (str, optional): Path to save the record table.
        log_file (str, optional): Path to a log file for this featurization.
        output_model_dataset_path (str, optional): Path to save the model-ready table.
        scaler_save_path (str, optional): Path to save the fitted scaler.

    Returns:
        list: Data objects with solvent_x, y, smiles, solvent_smiles, redox_type,
        data_type and solvent_id.
    """
    if mol_featurizer is None:
        raise ValueError("mol_featurizer must be provided")
    if path_to_sdf is None and molecules is None:
        raise ValueError("Either 'path_to_sdf' or 'molecules' must be provided")
    if path_to_sdf is not None and molecules is not None:
        raise ValueError("Provide only one of 'path_to_sdf' or 'molecules'")

    file_handler = None
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(file_handler)

    if molecules is None:
        molecules = [mol for mol in Chem.SDMolSupplier(path_to_sdf) if mol is not None]
        logger.info("Loaded %d molecules from %s", len(molecules), path_to_sdf)

    solvent_db, name_to_smiles = load_solvent_db(solvent_db_path)

    mol_features = [mol_featurizer.featurize(m) for m in molecules]

    # (mol_idx, E, solvent_smiles, redox_type, data_type)
    records = []
    logger.info("Processing %d molecules", len(molecules))
    for mol_idx, mol in enumerate(molecules):
        record = read_redox_record(mol, mol_idx, solvent_db, name_to_smiles)
        if record is None:
            continue
        records.append((mol_idx, *record))

    unique_solvents = sorted({solvent_smiles for _, _, solvent_smiles, _, _ in records})
    solvent_to_index = {smiles: idx for idx, smiles in enumerate(unique_solvents)}
    logger.info("Data points: %d, unique solvents: %d", len(records), len(unique_solvents))

    if scale_solvent:
        scaler = StandardScaler()
        scaled = scaler.fit_transform([solvent_db[s] for s in unique_solvents])
        solvent_to_descriptors = {s: scaled[i].tolist() for i, s in enumerate(unique_solvents)}
        logger.info("Scaler mean: %s, scale: %s", scaler.mean_, scaler.scale_)
        if scaler_save_path:
            joblib.dump(scaler, scaler_save_path)
            logger.info("Scaler saved to %s", scaler_save_path)
    else:
        solvent_to_descriptors = {s: solvent_db[s] for s in unique_solvents}
    if SOLVENT_DESCRIPTORS_OFF:
        solvent_to_descriptors = {s: [0.0] * len(SOLVENT_DESCRIPTOR_COLUMNS) for s in unique_solvents}
        logger.info("Solvent descriptors zeroed (ablation without solvent input)")

    all_data = []
    dataset_records = []
    model_dataset_records = []
    descriptor_names = list(SOLVENT_DESCRIPTOR_COLUMNS.values())

    for mol_idx, E, solvent_smiles, redox_type, data_type in records:
        redox_type_binary = 1 if redox_type == "ox" else 0
        target_type = 1 if data_type == "exp" else 0
        solvent_vector = solvent_to_descriptors[solvent_smiles] + [redox_type_binary, target_type]
        mol_smiles = Chem.MolToSmiles(molecules[mol_idx])

        features = copy.deepcopy(mol_features[mol_idx])
        features.solvent_smiles = solvent_smiles
        features.solvent_x = torch.tensor(solvent_vector, dtype=torch.float32).unsqueeze(0)
        features.y = {"E": torch.tensor([[E]])}
        features.smiles = mol_smiles
        features.redox_type = redox_type
        features.data_type = data_type
        features.solvent_id = torch.tensor([solvent_to_index[solvent_smiles]], dtype=torch.long)
        all_data.append(features)

        dataset_records.append({
            "molecule_smiles": mol_smiles,
            "solvent_smiles": solvent_smiles,
            "redox_type": redox_type,
            "E_value": E,
            "data_type": data_type,
            **dict(zip(descriptor_names, solvent_db[solvent_smiles])),
            "redox_type_binary": redox_type_binary,
            "target_type_indicator": target_type,
        })
        model_dataset_records.append({
            "molecule_smiles": mol_smiles,
            "solvent_smiles": solvent_smiles,
            "E_value": E,
            "data_type": data_type,
            "redox_type": redox_type,
            "solvent_vector": solvent_vector,
            **{f"{name}_norm": value for name, value in zip(descriptor_names, solvent_vector)},
            "redox_type_binary": solvent_vector[-2],
            "target_type_indicator": solvent_vector[-1],
            "scaled": scale_solvent,
        })

    if output_dataset_path:
        df = save_table(dataset_records, output_dataset_path)
        logger.info(
            "Dataset summary: %d records, %d calc, %d exp, %d ox, %d red, %d unique molecules, %d unique solvents",
            len(df), (df["data_type"] == "calc").sum(), (df["data_type"] == "exp").sum(),
            (df["redox_type"] == "ox").sum(), (df["redox_type"] == "red").sum(),
            df["molecule_smiles"].nunique(), df["solvent_smiles"].nunique(),
        )
    if output_model_dataset_path:
        save_table(model_dataset_records, output_model_dataset_path)

    if shuffle:
        random.Random(seed).shuffle(all_data)

    if file_handler is not None:
        logger.removeHandler(file_handler)
        file_handler.close()

    return all_data