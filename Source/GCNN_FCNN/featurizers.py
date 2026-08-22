import sys
import copy
import json
import random
import warnings
import pandas as pd
from typing import Union, Optional
import numpy as np

import dgl
import torch
from dgllife.utils import mol_to_bigraph
from rdkit import Chem
from rdkit.Chem import Mol
from torch_geometric.utils import from_networkx
from sklearn.preprocessing import StandardScaler
from Source.GCNN_FCNN.old_featurizer import ConvMolFeaturizer
from Source.config import ROOT_DIR

from rdkit.Chem import MACCSkeys, AllChem, DataStructs, Descriptors, rdFingerprintGenerator
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from Data.solubility.solvent_properties import solvent_data


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


DEFAULT_SOLVENT_DB_PATH = None 
DEFAULT_OUTPUT_DATASET_PATH = None
DEFAULT_LOG_FILE = None
DEFAULT_OUTPUT_MODEL_DATASET_PATH = None

SOLVENT_NAME_TO_SMILES = {
    'acetonitrile': 'CC#N',
    'water': 'O',
    'dimethylsulfoxide': 'CS(C)=O',
    'dimethylformamide': 'CN(C)C=O',
    'tetrahydrofuran': 'C1CCOC1',
    'dichloromethane': 'ClCCl',
    'benzonitrile': 'N#Cc1ccccc1',
    'dimethylacetamide': 'CC(=O)N(C)C',
    'methanol': 'CO',
    'ethanol': 'CCO',
    'acetone': 'CC(C)=O',
    'pyridine': 'c1ccncc1',
    'formamide': 'NC=O',
    '2-propanol': 'CC(C)O',
    'isopropanol': 'CC(C)O',
    'n-methylpyrrolidone': 'CN1CCCC1=O',
    'propylene carbonate': 'CC1COC(=O)O1',
    'dimethoxyethane': 'COCCOC',
    'hexamethylphosphoramide': 'CN(C)P(=O)(N(C)C)N(C)C',
    'dmso': 'CS(C)=O',
    'dmf': 'CN(C)C=O',
    'thf': 'C1CCOC1',
    'dcm': 'ClCCl',
    'meoh': 'CO',
    'etoh': 'CCO',
    'acn': 'CC#N',
    'nmp': 'CN1CCCC1=O',
}


def normalize_solvent_name(name):
    return name.lower().strip().replace(' ', '').replace('-', '')


def get_solvent_smiles(solvent_identifier):
    if not solvent_identifier or solvent_identifier.strip() == '':
        return None
    solvent_identifier = solvent_identifier.strip()
    mol = Chem.MolFromSmiles(solvent_identifier)
    if mol is not None:
        return Chem.MolToSmiles(mol)
    normalized = normalize_solvent_name(solvent_identifier)
    
    for name, smiles in SOLVENT_NAME_TO_SMILES.items():
        if normalize_solvent_name(name) == normalized:
            return smiles
    return None


def featurize_sdf_with_solvent_all(
    path_to_sdf=None,
    molecules=None,
    mol_featurizer=None,
    seed=42,
    shuffle=True,
    solvent_mode="descriptors",
    scale_solvent=True,
    solvent_db_path=DEFAULT_SOLVENT_DB_PATH,
    output_dataset_path=DEFAULT_OUTPUT_DATASET_PATH,
    log_file=DEFAULT_LOG_FILE,
    output_model_dataset_path=DEFAULT_OUTPUT_MODEL_DATASET_PATH,
    scaler_save_path=None,
    calibration_coefficients_path=None,
    excluded_lf_pairs=None):
    """
    Featurize molecules with solvent vector and redox_type from SDF data.
    
    NEW STRUCTURE: Handles solvent_red/ox and solvent_smiles_red/ox properties.
    Prioritizes E_calc values over E_exp values when both are present.
    Adds 'data_type' ('calc' or 'exp') and 'solvent_id' (as numeric index) attributes.

    Args:
        path_to_sdf (str, optional): Path to SDF file.
        molecules (list, optional): List of RDKit molecules.
        mol_featurizer (MolFeaturizer): Required mol_featurizer instance.
        seed (int): Random seed for shuffling.
        shuffle (bool): Shuffle output dataset.
        solvent_mode (str): One of {"descriptors", "onehot", "ones"}.
        scale_solvent (bool): Apply StandardScaler to solvent descriptors.
        solvent_db_path (str, optional): Path to solvent database Excel file.
        output_dataset_path (str, optional): Path to save dataset as CSV file.
        log_file (str, optional): Path to save log output.
        output_model_dataset_path (str, optional): Path to save model-ready dataset.

    Returns:
        list: List of data objects with solvent_x, y, data_type, and solvent_id.
    """
    if mol_featurizer is None:
        raise ValueError("mol_featurizer must be provided")

    # Load per-(solvent, redox_type) calibration coefficients for on-the-fly E_lf calculation.
    # Key: (canonical_solvent_smiles, redox_type) -> {"a": float, "b": float}
    _calibration = {}
    if calibration_coefficients_path:
        try:
            cal_df = pd.read_csv(calibration_coefficients_path)
            for _, row in cal_df.iterrows():
                if str(row.get("source", "")) == "global_fallback":
                    continue
                smiles = get_solvent_smiles(str(row["solvent"]))
                if smiles:
                    rt = "ox" if str(row["redox_type"]).lower() == "oxidation" else "red"
                    _calibration[(smiles, rt)] = {"a": float(row["a"]), "b": float(row["b"])}
        except Exception as e:
            print(f"Warning: Could not load calibration coefficients: {e}")

    _excluded = excluded_lf_pairs or set()

    # Set up logging to file if specified
    original_stdout = sys.stdout
    log_file_handle = None

    if log_file:
        try:
            log_file_handle = open(log_file, 'w', encoding='utf-8')
            sys.stdout = log_file_handle
        except Exception as e:
            print(f"Warning: Could not open log file {log_file}: {e}")
            log_file_handle = None

    try:
        # Load molecules
        if path_to_sdf is None and molecules is None:
            raise ValueError("Either 'path_to_sdf' or 'molecules' must be provided")
        if path_to_sdf is not None and molecules is not None:
            raise ValueError("Provide only one of 'path_to_sdf' or 'molecules'")

        if molecules is None:
            molecules = [mol for mol in Chem.SDMolSupplier(path_to_sdf) if mol is not None]
            print(f"Loaded {len(molecules)} molecules from {path_to_sdf}")

        solvent_db = {}
        if solvent_db_path:
            try:
                df = pd.read_csv(solvent_db_path)
                for _, row in df.iterrows():
                    smiles = str(row['smiles']).strip()
                    props = {
                        "Molecular weight": float(row['Molecular weight']) if pd.notna(row['Molecular weight']) else 0.0,
                        "Vapor pressure": float(row['Vapor pressure, mmHg(20C)']) if pd.notna(row['Vapor pressure, mmHg(20C)']) else 0.0,
                        "Flash point": float(row['Flash point, K']) if pd.notna(row['Flash point, K']) else 0.0,
                        "Refractive index": float(row['Refractive index, n20/D']) if pd.notna(row['Refractive index, n20/D']) else 0.0,
                        "Boiling point": float(row['Boiling point, C']) if pd.notna(row['Boiling point, C']) else 0.0,
                        "Melting point": float(row['Melting point, C']) if pd.notna(row['Melting point, C']) else 0.0,
                        "Viscosity": float(row['Viscosity, cp(20C)']) if pd.notna(row['Viscosity, cp(20C)']) else 0.0,
                        "Density": float(row['Density, g/mL(25C)']) if pd.notna(row['Density, g/mL(25C)']) else 0.0,
                        "Dielectric permittivity": float(row['Dielectric permittivity']) if pd.notna(row['Dielectric permittivity']) else 0.0
                    }
                    solvent_db[smiles] = props
                print(f"Loaded {len(solvent_db)} solvents from database: {solvent_db_path}")
            except Exception as e:
                print(f"Warning: Could not load solvent database from {solvent_db_path}: {e}")

        # Featurize molecules
        mol_features = [mol_featurizer.featurize(m) for m in molecules]
        all_data = []

        # Track unique solvents for numeric indexing
        unique_solvents = set()

        # Collect all solvent descriptor vectors and info
        solvent_desc_list = []
        solvent_info_list = []  # (mol_idx, E, solvent_smiles, target_type, redox_type, descriptor_vector, data_type_label)

        # For dataset export
        dataset_records = []
        # For model-ready dataset export
        model_dataset_records = []

        print(f"\nProcessing {len(molecules)} molecules...")
        
        for mol_idx, mol in enumerate(molecules):
            mol_props = mol.GetPropNames()
            
            # Read redox_type (required)
            if "redox_type" not in mol_props:
                print(f"Molecule {mol_idx+1}: No redox_type, skipping")
                continue
            
            redox_type_full = mol.GetProp("redox_type").strip().lower()
            if redox_type_full == "oxidation":
                redox_type = "ox"
            elif redox_type_full == "reduction":
                redox_type = "red"
            else:
                print(f"Molecule {mol_idx+1}: Invalid redox_type '{redox_type_full}', skipping")
                continue
            
            print(f"Molecule {mol_idx+1}/{len(molecules)}: {Chem.MolToSmiles(mol)}, redox_type={redox_type}")
            
            # Determine properties based on redox_type
            if redox_type == "ox":
                solvent_name_prop = "solvent_ox"
                solvent_smiles_prop = "solvent_smiles_ox"
            else:  # red
                solvent_name_prop = "solvent_red"
                solvent_smiles_prop = "solvent_smiles_red"
            
            # Get solvent SMILES (priority: solvent_smiles_red/ox > solvent_red/ox)
            solvent_smiles = None
            
            # Priority 1: solvent_smiles_red/ox
            if solvent_smiles_prop in mol_props:
                solvent_smiles_raw = mol.GetProp(solvent_smiles_prop).strip()
                solvent_smiles = get_solvent_smiles(solvent_smiles_raw)
                if solvent_smiles:
                    print(f"  Using solvent SMILES from {solvent_smiles_prop}: {solvent_smiles}")
            
            # Priority 2: solvent_red/ox (name)
            if solvent_smiles is None and solvent_name_prop in mol_props:
                solvent_name = mol.GetProp(solvent_name_prop).strip()
                solvent_smiles = get_solvent_smiles(solvent_name)
                if solvent_smiles:
                    print(f"  Converted solvent name '{solvent_name}' to SMILES: {solvent_smiles}")
            
            if solvent_smiles is None:
                print(f"  Could not determine solvent SMILES for {redox_type}, skipping")
                continue
            
            # Get E value.
            # For calc entries: prefer E_{redox_type}_V_calc_lf (pre-calibrated in SDF).
            # If absent, fall back to E_{redox_type}_V_calc and apply on-the-fly calibration
            # when coefficients are available. E_{redox_type}_V (exp) is used as last resort.
            lf_prop   = f"E_{redox_type}_V_calc_lf"
            calc_prop = f"E_{redox_type}_V_calc"
            exp_prop  = f"E_{redox_type}_V"

            E = None
            target_type = None
            target_source = None
            data_type_label = None

            if lf_prop in mol_props:
                try:
                    E = float(mol.GetProp(lf_prop))
                    target_type = 0
                    target_source = "calculated"
                    data_type_label = "calc"
                except ValueError:
                    print(f"  Warning: Could not parse {lf_prop}")

            if E is None and calc_prop in mol_props:
                try:
                    E_raw = float(mol.GetProp(calc_prop))
                    coef = _calibration.get((solvent_smiles, redox_type))
                    E = coef["a"] * E_raw + coef["b"] if coef else E_raw
                    target_type = 0
                    target_source = "calculated"
                    data_type_label = "calc"
                except ValueError:
                    print(f"  Warning: Could not parse {calc_prop}")

            if E is None and exp_prop in mol_props:
                try:
                    E = float(mol.GetProp(exp_prop))
                    target_type = 1
                    target_source = "experimental"
                    data_type_label = "exp"
                except ValueError:
                    print(f"  Warning: Could not parse {exp_prop}")

            if E is None:
                print(f"  No E value found for {redox_type}, skipping")
                continue

            # Drop LF entries for (solvent, redox_type) pairs with poor LF-exp correlation.
            if data_type_label == "calc" and (solvent_smiles, redox_type) in _excluded:
                continue
            
            # Create solvent descriptors
            if solvent_mode == "descriptors":
                # Try to get solvent properties from database
                if solvent_smiles in solvent_db:
                    solvent_props = solvent_db[solvent_smiles]
                    desc = [
                        #solvent_props["Molecular weight"],
                        solvent_props["Vapor pressure"],
                        #solvent_props["Flash point"],
                        #solvent_props["Refractive index"],
                        #solvent_props["Boiling point"],
                        solvent_props["Melting point"],
                        solvent_props["Viscosity"],
                        solvent_props["Density"],
                        solvent_props["Dielectric permittivity"]
                    ]
                    print(f"    Using database properties")
                else:
                    # Fallback to RDKit descriptors
                    print(f"    Solvent not in database, using RDKit descriptors")
                    solvent_mol = Chem.MolFromSmiles(solvent_smiles)
                
                # Add target_type and redox_type_binary
                redox_type_idx = 1 if redox_type == "ox" else 0
                desc_with_target_and_redox = desc + [target_type, redox_type_idx]
                solvent_desc_list.append(desc_with_target_and_redox)
                solvent_info_list.append((mol_idx, E, solvent_smiles, target_type, redox_type, desc_with_target_and_redox, data_type_label))
                
                # Add to dataset records
                record = {
                    'molecule_smiles': Chem.MolToSmiles(molecules[mol_idx]),
                    'solvent_smiles': solvent_smiles,
                    'redox_type': redox_type,
                    'E_value': E,
                    'target_type': target_source,
                    'data_type': data_type_label,
                    'solvent_mode': solvent_mode,
                    #'mol_weight': desc[0],
                    'vapor_pressure': desc[0],
                    #'flash_point': desc[2],
                    #'refractive_index': desc[3],
                    #'boiling_point': desc[4],
                    'melting_point': desc[1],
                    'viscosity': desc[2],
                    'density': desc[3],
                    'dielectric': desc[4],
                    'target_type_indicator': target_type,
                    'redox_type_binary': redox_type_idx
                }
                dataset_records.append(record)
            else:
                # For non-descriptor modes
                redox_type_idx = 1 if redox_type == "ox" else 0
                solvent_info_list.append((mol_idx, E, solvent_smiles, target_type, redox_type, None, data_type_label))
                
                record = {
                    'molecule_smiles': Chem.MolToSmiles(molecules[mol_idx]),
                    'solvent_smiles': solvent_smiles,
                    'redox_type': redox_type,
                    'E_value': E,
                    'target_type': target_source,
                    'data_type': data_type_label,
                    'solvent_mode': solvent_mode,
                    #'mol_weight': 0,
                    'vapor_pressure': 0,
                    #'refractive_index': 0,
                    #'boiling_point': 0,
                    'melting_point': 0,
                    'density': 0,
                    'dielectric': 0,
                    'target_type_indicator': target_type,
                    'redox_type_binary': redox_type_idx
                }
                dataset_records.append(record)
            
            # Add to unique solvents set
            unique_solvents.add(solvent_smiles)
        
        print(f"\nTotal data points processed: {len(solvent_info_list)}")
        print(f"Solvent descriptor vectors: {len(solvent_desc_list)}")
        print(f"Unique solvents found: {len(unique_solvents)}")
        
        # Create solvent mappings
        unique_solvent_list = sorted(list(unique_solvents))
        solvent_to_index = {smiles: idx for idx, smiles in enumerate(unique_solvent_list)}
        print(f"Solvent index mapping created for {len(solvent_to_index)} solvents")
        
        # Fit scaler if needed
        scaler = None
        scaled_desc = []
        if solvent_mode == "descriptors" and scale_solvent and len(solvent_desc_list) > 0:
            scaler = StandardScaler()
            
            # Collect UNIQUE descriptors for each solvent
            unique_solvent_descriptors = {}
            solvent_descriptor_list = []
            
            for info in solvent_info_list:
                mol_idx, E, solvent_smiles, target_type, redox_type, desc, data_type_label = info
                if solvent_smiles not in unique_solvent_descriptors:
                    unique_solvent_descriptors[solvent_smiles] = desc[:-2]  # without target_type and redox_type
                    solvent_descriptor_list.append(desc[:-2])
            
            # Fit scaler on unique solvents
            scaled_unique_descriptors = scaler.fit_transform(solvent_descriptor_list)
            if scaler_save_path:
                import joblib
                from pathlib import Path
                Path(scaler_save_path).parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(scaler, scaler_save_path)
                print(f"Scaler saved -> {scaler_save_path}")
            # Create mapping: solvent -> normalized descriptors
            solvent_to_scaled = {}
            for i, solvent_smiles in enumerate(list(unique_solvent_descriptors.keys())):
                solvent_to_scaled[solvent_smiles] = list(scaled_unique_descriptors[i])
            
            # Create scaled_desc for all records
            for info in solvent_info_list:
                mol_idx, E, solvent_smiles, target_type, redox_type, desc, data_type_label = info
                scaled_vector = solvent_to_scaled[solvent_smiles] + [desc[-2], desc[-1]]  # add target_type and redox_type
                scaled_desc.append(scaled_vector)
            
            print(f"\nScaled solvent descriptors")
            print(f"Scaler mean: {scaler.mean_}")
            print(f"Scaler scale: {scaler.scale_}")
            print(f"Number of unique solvents: {len(unique_solvent_descriptors)}")
        else:
            scaled_desc = solvent_desc_list
        
        # Create feature objects
        desc_index = 0
        print(f"\nCreating feature objects...")
        
        for info in solvent_info_list:
            mol_idx, E, solvent_smiles, target_type, redox_type, desc, data_type_label = info
            
            features = copy.deepcopy(mol_features[mol_idx])
            features.solvent_smiles = solvent_smiles
            
            if solvent_mode == "descriptors":
                solvent_vector = scaled_desc[desc_index]
                features.solvent_x = torch.tensor(solvent_vector, dtype=torch.float32).unsqueeze(0)
                
                # Save model-ready data
                model_record = {
                    'molecule_smiles': Chem.MolToSmiles(molecules[mol_idx]),
                    'solvent_smiles': solvent_smiles,
                    'E_value': E,
                    'target_type': 'calculated' if data_type_label == 'calc' else 'experimental',
                    'data_type': data_type_label,
                    'redox_type': redox_type,
                    'solvent_vector': solvent_vector,
                    #'mol_weight_norm': solvent_vector[0],
                    'vapor_pressure_norm': solvent_vector[0],
                    #'refractive_index_norm': solvent_vector[2],
                    #'boiling_point_norm': solvent_vector[3],
                    'melting_point_norm': solvent_vector[1],
                    'viscosity_norm': solvent_vector[2],
                    'density_norm': solvent_vector[3],
                    'dielectric_norm': solvent_vector[4],
                    'target_type_indicator': solvent_vector[5],
                    'redox_type_binary': solvent_vector[6],
                    'solvent_mode': solvent_mode,
                    'scaled': scale_solvent
                }
                model_dataset_records.append(model_record)
                desc_index += 1
                
            elif solvent_mode == "onehot":
                # One-hot encoding (would need implementation)
                print(f"Warning: onehot mode not fully implemented, using placeholder")
                redox_type_idx = 1 if redox_type == "ox" else 0
                vector = torch.zeros(len(unique_solvents) + 2)  # +2 for target_type and redox
                vector[solvent_to_index[solvent_smiles]] = 1.0
                vector[-2] = target_type
                vector[-1] = redox_type_idx
                features.solvent_x = vector.unsqueeze(0)
                
            elif solvent_mode == "ones":
                ones_with_target = torch.ones(7, dtype=torch.float32)
                target_type_tensor = torch.tensor([target_type], dtype=torch.float32)
                redox_type_idx = 1 if redox_type == "ox" else 0
                redox_type_tensor = torch.tensor([redox_type_idx], dtype=torch.float32)
                ones_with_target_and_redox = torch.cat([ones_with_target, target_type_tensor, redox_type_tensor])
                features.solvent_x = ones_with_target_and_redox.unsqueeze(0)
            else:
                raise ValueError(f"Unknown solvent_mode: {solvent_mode}")
            
            # Set y value
            features.y = {"E": torch.tensor([[E]])}
            
            # Additional attributes
            features.smiles = Chem.MolToSmiles(molecules[mol_idx])
            features.redox_type = redox_type
            features.data_type = data_type_label  # 'calc' or 'exp'
            features.solvent_id = torch.tensor([solvent_to_index[solvent_smiles]], dtype=torch.long)
            
            all_data.append(features)
        
        print(f"\nCreated {len(all_data)} data points")
        
        # Save dataset to file
        if output_dataset_path and dataset_records:
            dataset_df = pd.DataFrame(dataset_records)
            try:
                if str(output_dataset_path).endswith('.csv'):
                    dataset_df.to_csv(output_dataset_path, index=False)
                elif str(output_dataset_path).endswith('.xlsx'):
                    dataset_df.to_excel(output_dataset_path, index=False)
                else:
                    dataset_df.to_csv(str(output_dataset_path) + '.csv', index=False)
                
                print(f"\nDataset saved to {output_dataset_path}")
                print(f"\n=== DATASET SUMMARY ===")
                print(f"Total records: {len(dataset_df)}")
                print(f"Calculated targets: {len(dataset_df[dataset_df['data_type'] == 'calc'])}")
                print(f"Experimental targets: {len(dataset_df[dataset_df['data_type'] == 'exp'])}")
                print(f"Oxidation records: {len(dataset_df[dataset_df['redox_type'] == 'ox'])}")
                print(f"Reduction records: {len(dataset_df[dataset_df['redox_type'] == 'red'])}")
                print(f"Unique molecules: {dataset_df['molecule_smiles'].nunique()}")
                print(f"Unique solvents: {dataset_df['solvent_smiles'].nunique()}")
                
            except Exception as e:
                print(f"Warning: Could not save dataset: {e}")
        
        # Save model-ready dataset
        if output_model_dataset_path and model_dataset_records:
            model_dataset_df = pd.DataFrame(model_dataset_records)
            try:
                if str(output_model_dataset_path).endswith('.csv'):
                    model_dataset_df.to_csv(output_model_dataset_path, index=False)
                elif str(output_model_dataset_path).endswith('.xlsx'):
                    model_dataset_df.to_excel(output_model_dataset_path, index=False)
                else:
                    model_dataset_df.to_csv(str(output_model_dataset_path) + '.csv', index=False)
                
                print(f"\nModel-ready dataset saved to {output_model_dataset_path}")
                
            except Exception as e:
                print(f"Warning: Could not save model dataset: {e}")
        
        # Shuffle if requested
        if shuffle:
            print(f"\nShuffling data with seed {seed}")
            random.Random(seed).shuffle(all_data)
        
        print(f"\n=== PROCESSING COMPLETED ===")
        return all_data
    
    finally:
        # Restore stdout
        if log_file_handle:
            sys.stdout = original_stdout
            log_file_handle.close()
            print(f"Log output saved to: {log_file}")

def get_ecfc(molecule):
    ecfc_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize = 1024)
    ecfc_array = np.zeros((1024,), dtype = int)
    ecfc = ecfc_gen.GetCountFingerprint(molecule)
    DataStructs.ConvertToNumpyArray(ecfc, ecfc_array)
    return ecfc_array

def get_solvent_vector(str):
    """"
    Implements a vector representation of the solvent
    based on MACCS keys and rdkit descriptors

    """
    
    names_dict = {'acetyl acetate': 'CC(=O)OC(C)=O', 'chloroform': 'ClC(Cl)Cl', 'n-butanol': 'CCCCO',
                  'n-heptane': 'CCCCCCC', 'γ-butyrolactone': 'O=C1CCCO1', 'DMAC': 'CN(C)C(C)=O',
                  'isooctanol': 'CC(C)CCCCCO', 'isopropyl acetate': 'CC(C)OC(C)=O', 'n-pentanol': 'CCCCCO',
                  'ethyl formate': 'CCOC=O', 'acetylacetone': 'CC(=O)CC(C)=O', 'n-decanol': 'CCCCCCCCCCO',
                  'trichloroethylene': 'ClC=C(Cl)Cl', '1,1-dichloroethane': 'CC(Cl)Cl', 'n-propanol': 'CCCO',
                  'p-tert-butyltoluene': 'Cc1ccc(cc1)C(C)(C)C', '1,4-dioxane': 'C1COCCO1', 'acetone': 'CC(C)=O',
                  'anisole': 'COc1ccccc1', '1,2-dichlorobenzene': 'Clc1ccccc1Cl', 'tetrachloromethane': 'ClC(Cl)(Cl)Cl',
                  'n-heptanol': 'CCCCCCCO', 'sulfolane': 'O=S]1(=O)CCCC1', 'epichlorohydrin': 'ClCC1CO1',
                  'methylcyclohexane': 'CC1CCCCC1', '2-aminoethanol': 'NCCO', 'sec-butanol': 'CCC(C)O',
                  'tributyl phosphate': 'CCCCO[P(OCCCC)OCCCC', '3,6-dioxa-1-decanol': 'CCCCOCCOCCO',
                  '2-hexanone': 'CCCCC(C)=O', 'n-hexanol': 'CCCCCCO', 'pyridine': 'c1ccncc1', 'n-octane': 'CCCCCCCC',
                  'benzylalcohol': 'OCc1ccccc1', 'MIBK': 'CC(C)CC(C)=O', 'aniline': 'Nc1ccccc1', 'ethanol': 'CCO',
                  'n-pentane': 'CCCCC', 'morpholine-4-carbaldehyde': 'O=CN1CCOCC1', '2-butanone': 'CCC(C)=O',
                  'DEF': 'Fe]|1|2|3|4|5|ON(C)C(=O|1)CCC(=O)NCCCCCN(O|2)C(=O|3)CCC(=O)NCCCCCN(O|4)C(=O|5)C',
                  'NMP': 'CN1CCCC1=O', 'decalin': 'C1CCC2CCCCC2C1', '2-pentanol': 'CCCC(C)O', 'transcutol': 'CCOCCOCCO',
                  'sec-butyl acetate': 'CCC(C)OC(C)=O', '1-methoxy-2-propyl acetate': 'COCC(C)OC(C)=O',
                  '2-(2-methoxypropoxy) propanol': 'COC(C)COC(C)CO', 'cyclohexane': 'C1CCCCC1',
                  'triethyl phosphate': 'CCO[P(OCC)OCC', 'DMF': 'CN(C)C=O', 'methyl acetate': 'COC(C)=O',
                  'dichloromethane': 'ClCCl', 'tert-butanol': 'CC(C)(C)O', 'acetic acid': 'CC(O)=O',
                  'ethylene glycol': 'OCCO', 'acetonitrile': 'CC#N', 'DMSO': 'CS=O', '2-propoxyethanol': 'CCCOCCO',
                  'n-dodecane': 'CCCCCCCCCCCC', 'formamide': 'NC=O', 'THF': 'C1CCOC1', 'methanol': 'CO',
                  'isopropanol': 'CC(C)O', 'formic acid': 'OC=O', 'water': 'O', 'propionic acid': 'CCC(O)=O',
                  'dipropyl ether': 'CCCOCCC', 'diisopropyl ether': 'CC(C)OC(C)C', '2-methoxyethanol': 'COCCO',
                  'ethyl acetate': 'CCOC(C)=O', 'benzene': 'c1ccccc1', 'p-xylene': 'Cc1ccc(C)cc1',
                  '2-octanol': 'CCCCCCC(C)O', 'MTBE': 'COC(C)(C)C', '1,2-dichloroethane': 'ClCCCl',
                  'ethyl benzene': 'CCc1ccccc1', '2-ethoxyethanol': 'CCOCCO', 'n-propyl acetate': 'CCCOC(C)=O',
                  'methyl propionate': 'CCC(=O)OC', 'triethyl orthoformate': 'CCOC(OCC)OCC',
                  '1-propoxy-2-propanol': 'CCCOCC(C)O', 'diacetone alcohol': 'CC(=O)CC(C)(C)O',
                  'benzyl alcohol': 'OCc1ccccc1', '2-butoxyethanol': 'CCCCOCCO', 'propylene carbonate': 'CC1COC(=O)O1',
                  'nonan-1-ol': 'CCCCCCCCCO', 'sec-pentanol': 'CCCC(C)O', '2-isopropoxyethanol': 'CC(C)OCCO',
                  'DMS': 'CS=O', 'diethylene glycol': 'OCCOCCO', 'MEK': 'CCC(C)=O', 'n-pentyl acetate': 'CCCCCOC(C)=O',
                  'cyclohexanone': 'O=C1CCCCC1', 'tert-amyl alcohol': 'CCC(C)(C)O', 'cyclopentanone': 'O=C1CCCC1',
                  'morpholine': 'C1COCCN1', 'm-xylene': 'Cc1cccc(C)c1', '3-pentanone': 'CCC(=O)CC',
                  '2-ethylhexanol': 'CCCCC(CC)CO', 'n-hexane': 'CCCCCC', 'isooctane': 'CCCCCC(C)C',
                  'isobutanol': 'CC(C)CO', 'isopentanol': 'CC(C)CCO', 'n-hexadecane': 'CCCCCCCCCCCCCCCC',
                  'diethyl ether': 'CCOCC', 'ethylbenzene': 'CCc1ccccc1', 'n-butyric acid': 'CCCC(O)=O',
                  'acrylic acid': 'OC(=O)C=C', '2-ethyl-n-hexanol': 'CCCCC(CC)CO', 'chlorobenzene': 'Clc1ccccc1',
                  '3-oxa-1,5-pentanediol': 'OCCOCCO', '1,3-propanediol': 'OCCCO', 'DMA': 'CC(C)=CCOP(=O)OP(O)=O',
                  'n-dodecanol': 'CCCCCCCCCCCCO', 'n-butyl acetate': 'CCCCOC(C)=O',
                  'methyl 4-tert-butylbenzoate': 'COC(=O)c1ccc(cc1)C(C)(C)C', 'isobutyl acetate': 'CC(C)COC(C)=O',
                  'o-xylene': 'Cc1ccccc1C', '1-methoxy-2-propanol': 'COCC(C)O',
                  '1,2,4-trichlorobenzene': 'Clc1ccc(Cl)c(Cl)c1', 'toluene': 'Cc1ccccc1', 'methyl formate': 'COC=O',
                  'tert-butyl acetate': 'CC(=O)OC(C)(C)C', 'n-octanol': 'CCCCCCCCO', 'propylene glycol': 'CC(O)CO',
                  '4-methylpyridine': 'Cc1ccncc1', 'acetophenone': 'CC(=O)c1ccccc1', '2-pentanone': 'CCCC(C)=O',
                  '1-bromopropane': 'CCCBr', 'p-cymene': 'CC(C)c1ccc(C)cc1', 'cumene': 'CC(C)c1ccccc1',
                  '2-(2-butoxyethoxy)ethanol': 'CCCCOCCOCCO', '1,2-diethoxyethane': 'CCOCCOCC'}

    solvent_mol = Chem.MolFromSmiles(names_dict[str])
    if solvent_mol is not None:
        vector = []

        vector += [Chem.Descriptors.MaxAbsPartialCharge(solvent_mol)]
        vector += [Chem.EState.EState.MaxAbsEStateIndex(solvent_mol)]
        vector += [Chem.Descriptors.MinPartialCharge(solvent_mol)]
        vector += [Chem.EState.EState.MinAbsEStateIndex(solvent_mol)]
        vector += [Chem.Descriptors.MaxPartialCharge(solvent_mol)]
        vector += [Chem.EState.EState.MinEStateIndex(solvent_mol)]
        vector += [Chem.Descriptors.MinAbsPartialCharge(solvent_mol)]
        vector += [Chem.EState.EState.MaxEStateIndex(solvent_mol)]
        vector += [Chem.Crippen.MolMR(solvent_mol)]

        vector += [Chem.MolSurf.SlogP_VSA2(solvent_mol)]
        vector += [Chem.EState.EState_VSA.VSA_EState2(solvent_mol)]
        vector += [Chem.MolSurf.SlogP_VSA1(solvent_mol)]
        vector += [Chem.EState.EState_VSA.EState_VSA1(solvent_mol)]
        vector += [Chem.MolSurf.PEOE_VSA1(solvent_mol)]
        vector += [Chem.EState.EState_VSA.VSA_EState8(solvent_mol)]
        vector += [Chem.EState.EState_VSA.EState_VSA9(solvent_mol)]
        vector += [Chem.EState.EState_VSA.VSA_EState6(solvent_mol)]
        vector += [Chem.MolSurf.SMR_VSA3(solvent_mol)]
        vector += [Chem.EState.EState_VSA.VSA_EState1(solvent_mol)]
        vector += [Chem.EState.EState_VSA.VSA_EState3(solvent_mol)]
        vector += [Chem.MolSurf.PEOE_VSA8(solvent_mol)]
        vector += [Chem.MolSurf.SMR_VSA5(solvent_mol)]
        vector += [Chem.MolSurf.PEOE_VSA7(solvent_mol)]
        vector += [Chem.MolSurf.SMR_VSA7(solvent_mol)]
        vector += [Chem.EState.EState_VSA.VSA_EState9(solvent_mol)]
        vector += [Chem.MolSurf.SMR_VSA6(solvent_mol)]
        vector += [Chem.MolSurf.SMR_VSA1(solvent_mol)]
        vector += [Chem.MolSurf.PEOE_VSA3(solvent_mol)]
        vector += [Chem.MolSurf.SMR_VSA10(solvent_mol)]
        vector += [Chem.MolSurf.SlogP_VSA5(solvent_mol)]
        vector += [Chem.EState.EState_VSA.EState_VSA5(solvent_mol)]
        vector += [Chem.MolSurf.PEOE_VSA2(solvent_mol)]
        vector += [Chem.MolSurf.SlogP_VSA4(solvent_mol)]
        vector += [Chem.MolSurf.SlogP_VSA6(solvent_mol)]
        vector += [Chem.EState.EState_VSA.EState_VSA8(solvent_mol)]
        vector += [Chem.MolSurf.PEOE_VSA12(solvent_mol)]
        vector += [Chem.MolSurf.PEOE_VSA14(solvent_mol)]
        vector += [Chem.EState.EState_VSA.VSA_EState10(solvent_mol)]
        vector += [Chem.MolSurf.PEOE_VSA13(solvent_mol)]
        vector += [Chem.MolSurf.SlogP_VSA12(solvent_mol)]
        vector += [Chem.MolSurf.PEOE_VSA6(solvent_mol)]
        vector += [Chem.EState.EState_VSA.VSA_EState4(solvent_mol)]
        vector += [Chem.MolSurf.PEOE_VSA9(solvent_mol)]
        vector += [Chem.MolSurf.SlogP_VSA10(solvent_mol)]
        vector += [Chem.MolSurf.TPSA(solvent_mol)]

        vector += [Chem.GraphDescriptors.Chi1(solvent_mol)]
        vector += [Chem.GraphDescriptors.Chi1v(solvent_mol)]
        vector += [Chem.GraphDescriptors.Chi1n(solvent_mol)]
        vector += [Chem.GraphDescriptors.Chi4v(solvent_mol)]
        vector += [Chem.GraphDescriptors.Chi2v(solvent_mol)]
        vector += [Chem.GraphDescriptors.Chi3v(solvent_mol)]
        vector += [Chem.GraphDescriptors.Chi4n(solvent_mol)]
        vector += [Chem.GraphDescriptors.Chi0n(solvent_mol)]
        vector += [Chem.GraphDescriptors.Chi0(solvent_mol)]
        vector += [Chem.GraphDescriptors.Kappa1(solvent_mol)]
        vector += [Chem.GraphDescriptors.Chi0v(solvent_mol)]
        vector += [Chem.GraphDescriptors.Chi3n(solvent_mol)]
        vector += [Chem.GraphDescriptors.Kappa3(solvent_mol)]
        vector += [Chem.GraphDescriptors.Kappa2(solvent_mol)]
        vector += [Chem.GraphDescriptors.Chi2n(solvent_mol)]
        vector += [Chem.GraphDescriptors.BalabanJ(solvent_mol)]
        vector += [Chem.GraphDescriptors.HallKierAlpha(solvent_mol)]
        vector += [Chem.GraphDescriptors.BertzCT(solvent_mol)]
        vector += [Chem.Lipinski.HeavyAtomCount(solvent_mol)]
        vector += [Chem.Descriptors.MolWt(solvent_mol)]
        vector += [Chem.Lipinski.NHOHCount(solvent_mol)]
        vector += [Chem.Lipinski.NOCount(solvent_mol)]
        vector += [Chem.Crippen.MolLogP(solvent_mol)]
        vector += [Chem.Lipinski.NumHDonors(solvent_mol)]
        vector += [Chem.Lipinski.NumHeteroatoms(solvent_mol)]
        vector += [Chem.Lipinski.NumHAcceptors(solvent_mol)]
        vector += [Chem.Lipinski.NumRotatableBonds(solvent_mol)]
        vector += [Chem.Descriptors.NumValenceElectrons(solvent_mol)]
        vector += [Chem.Lipinski.NumAromaticCarbocycles(solvent_mol)]
        vector += [Chem.Lipinski.NumAromaticRings(solvent_mol)]
        vector += [Chem.Lipinski.NumAromaticHeterocycles(solvent_mol)]
        vector += [Chem.Lipinski.NumSaturatedHeterocycles(solvent_mol)]

        vector.extend(list(AllChem.GetMACCSKeysFingerprint(solvent_mol)))
        vector.extend(get_ecfc(solvent_mol))

        return vector
    else:
        raise ValueError(f"invalid molecule input type: {mol.GetProp('Solvent')}")

def featurize_sdf_with_solvent_united(path_to_sdf=None, molecules=None, mol_featurizer=ConvMolFeaturizer(),
                               seed=42, shuffle=True, include_descriptors='', test_solvents='', n_test_in_train=0):
    """"
    Extract molecules from .sdf file and featurize them

    Parameters
    ----------
    path_to_sdf : str
        path to .sdf file with data
        single molecule in .sdf file can contain properties like "logK_{metal}"
        each of these properties will be transformed into a different training sample
    molecules: List[Chem.Mol]
        list of molecules, can be used instead of path_to_sdf
    mol_featurizer : featurizer, optional
        instance of the class used for extracting features of organic molecule
    seed: int = 42
        random seed for reproducibility
    shuffle: bool
        whether to shuffle data after featurization
    include_descriptors: str
        string with additional descriptors separated by a space
    test_solvents: str
        string with solvents to be put in test part (separated by a space)
    n_test_in_train: int
        a number of test-solvent samples in train

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
    mol_features = [mol_featurizer.featurize(m) for m in mols]

    all_data_train, all_data_test = [], []
    my_counter=0

    for mol_ind in range(len(mols)):
        logS_list = []
        for target in [prop for prop in mols[mol_ind].GetPropNames() if prop.startswith("Solubility")]:
            logS_list += [float(mols[mol_ind].GetProp(target))]
            element_symbol = mols[mol_ind].GetProp('Solvent')
            vector = get_solvent_vector(element_symbol)

            #dG eps BP_mols BP_solvs
            if 'dG' in include_descriptors:
                vector += [float(mols[mol_ind].GetProp('dG'))]
            if 'eps' in include_descriptors:
                vector += [float(mols[mol_ind].GetProp('eps'))]
            if 'BP_mols' in include_descriptors:
                vector += [float(mols[mol_ind].GetProp('BP_mols'))]
            if 'BP_solvs' in include_descriptors:
                vector += [float(mols[mol_ind].GetProp('BP_solvs'))]
            
            vector = np.array(vector, dtype=np.float32)

        features = copy.deepcopy(mol_features[mol_ind])
        features.x_fully_connected = torch.tensor(vector).unsqueeze(0)
        features.y = {"Solubility": torch.tensor([logS_list])}
        if mols[mol_ind].GetProp('Solvent') in test_solvents:
            if my_counter<n_test_in_train:
                all_data_train += [features]
                my_counter += 1
            else:
                all_data_test += [features]
        else:
            all_data_train += [features]

    if shuffle:
        random.Random(seed).shuffle(all_data_test)
        random.Random(seed).shuffle(all_data_train)
        
    return all_data_train, all_data_test

def featurize_sdf_with_solvent_united_no_split(path_to_sdf=None, molecules=None, mol_featurizer=ConvMolFeaturizer(),
                               seed=42, shuffle=True, include_descriptors=''):
    """"
    Extract molecules from .sdf file and featurize them

    Parameters
    ----------
    path_to_sdf : str
        path to .sdf file with data
        single molecule in .sdf file can contain properties like "logK_{metal}"
        each of these properties will be transformed into a different training sample
    molecules: List[Chem.Mol]
        list of molecules, can be used instead of path_to_sdf
    mol_featurizer : featurizer, optional
        instance of the class used for extracting features of organic molecule
    seed: int = 42
        random seed for reproducibility
    shuffle: bool
        whether to shuffle data after featurization
    include_descriptors: str
        string with additional descriptors separated by a space

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
    mol_features = [mol_featurizer.featurize(m) for m in mols]

    all_data = []


    for mol_ind in range(len(mols)):
        logS_list = []
        for target in [prop for prop in mols[mol_ind].GetPropNames() if prop.startswith("LogS")]:
            logS_list += [float(mols[mol_ind].GetProp(target))]
            element_symbol = mols[mol_ind].GetProp('Solvent')
            vector = get_solvent_vector(element_symbol)

            #dG eps BP_mols BP_solvs
            if 'dG' in include_descriptors:
                vector += [float(mols[mol_ind].GetProp('dG'))]
            if 'eps' in include_descriptors:
                vector += [float(mols[mol_ind].GetProp('eps'))]
            if 'BP_mols' in include_descriptors:
                vector += [float(mols[mol_ind].GetProp('BP_mols'))]
            if 'BP_solvs' in include_descriptors:
                vector += [float(mols[mol_ind].GetProp('BP_solvs'))]
            
            vector = np.array(vector, dtype=np.float32)

        features = copy.deepcopy(mol_features[mol_ind])
        features.x_fully_connected = torch.tensor(vector).unsqueeze(0)
        features.y = {"Solubility": torch.tensor([logS_list])}
        all_data += [features]

    if shuffle: random.Random(seed).shuffle(all_data)


    return all_data

def featurize_sdf_with_solvent_experimental(path_to_sdf=None, molecules=None, mol_featurizer=ConvMolFeaturizer(),
                               seed=42, shuffle=True, use_maccskeys=False, use_exp_values=['MW', 'RI', 'eps', 'BP', 'd', 'MP', 'WS', 'FP']):
    """"
    Extract molecules from .sdf file and featurize them

    Parameters
    ----------
    path_to_sdf : str
        path to .sdf file with data
        single molecule in .sdf file can contain properties like "logK_{metal}"
        each of these properties will be transformed into a different training sample
    molecules: List[Chem.Mol]
        list of molecules, can be used instead of path_to_sdf
    mol_featurizer : featurizer, optional
        instance of the class used for extracting features of organic molecule
    seed: int = 42
        random seed for reproducibility
    shuffle: bool
        whether to shuffle data after featurization
    include_descriptors: str
        string with additional descriptors separated by a space

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
    mol_features = [mol_featurizer.featurize(m) for m in mols]

    all_data = []


    for mol_ind in range(len(mols)):
        logS_list = []
        for target in [prop for prop in mols[mol_ind].GetPropNames() if prop.startswith("LogS")]:
            logS_list += [float(mols[mol_ind].GetProp(target))]
            element_symbol = mols[mol_ind].GetProp('Solvent')
            vector = get_solvent_vector_experimental(element_symbol, use_maccskeys, use_exp_values)
            #print(f'LEN_VECTOR {len(vector)}')
            
            vector = np.array(vector, dtype=np.float32)

        features = copy.deepcopy(mol_features[mol_ind])
        features.x_fully_connected = torch.tensor(vector).unsqueeze(0)
        features.y = {"Solubility": torch.tensor([logS_list])}
        all_data += [features]

    if shuffle: random.Random(seed).shuffle(all_data)


    return all_data

def get_solvent_vector_experimental(str, use_maccskeys=False, use_exp_values=['MW', 'RI', 'eps', 'BP', 'd', 'MP', 'WS', 'FP']):
    """"
    Implements a vector representation of the solvent
    based on MACCS keys and experimental descriptors

    """
    
    names_dict = {'acetyl acetate': 'CC(=O)OC(C)=O', 'chloroform': 'ClC(Cl)Cl', 'n-butanol': 'CCCCO',
                  'n-heptane': 'CCCCCCC', 'γ-butyrolactone': 'O=C1CCCO1', 'DMAC': 'CN(C)C(C)=O',
                  'isooctanol': 'CC(C)CCCCCO', 'isopropyl acetate': 'CC(C)OC(C)=O', 'n-pentanol': 'CCCCCO',
                  'ethyl formate': 'CCOC=O', 'acetylacetone': 'CC(=O)CC(C)=O', 'n-decanol': 'CCCCCCCCCCO',
                  'trichloroethylene': 'ClC=C(Cl)Cl', '1,1-dichloroethane': 'CC(Cl)Cl', 'n-propanol': 'CCCO',
                  'p-tert-butyltoluene': 'Cc1ccc(cc1)C(C)(C)C', '1,4-dioxane': 'C1COCCO1', 'acetone': 'CC(C)=O',
                  'anisole': 'COc1ccccc1', '1,2-dichlorobenzene': 'Clc1ccccc1Cl', 'tetrachloromethane': 'ClC(Cl)(Cl)Cl',
                  'n-heptanol': 'CCCCCCCO', 'sulfolane': 'O=S]1(=O)CCCC1', 'epichlorohydrin': 'ClCC1CO1',
                  'methylcyclohexane': 'CC1CCCCC1', '2-aminoethanol': 'NCCO', 'sec-butanol': 'CCC(C)O',
                  'tributyl phosphate': 'CCCCO[P(OCCCC)OCCCC', '3,6-dioxa-1-decanol': 'CCCCOCCOCCO',
                  '2-hexanone': 'CCCCC(C)=O', 'n-hexanol': 'CCCCCCO', 'pyridine': 'c1ccncc1', 'n-octane': 'CCCCCCCC',
                  'benzylalcohol': 'OCc1ccccc1', 'MIBK': 'CC(C)CC(C)=O', 'aniline': 'Nc1ccccc1', 'ethanol': 'CCO',
                  'n-pentane': 'CCCCC', 'morpholine-4-carbaldehyde': 'O=CN1CCOCC1', '2-butanone': 'CCC(C)=O',
                  'DEF': 'Fe]|1|2|3|4|5|ON(C)C(=O|1)CCC(=O)NCCCCCN(O|2)C(=O|3)CCC(=O)NCCCCCN(O|4)C(=O|5)C',
                  'NMP': 'CN1CCCC1=O', 'decalin': 'C1CCC2CCCCC2C1', '2-pentanol': 'CCCC(C)O', 'transcutol': 'CCOCCOCCO',
                  'sec-butyl acetate': 'CCC(C)OC(C)=O', '1-methoxy-2-propyl acetate': 'COCC(C)OC(C)=O',
                  '2-(2-methoxypropoxy) propanol': 'COC(C)COC(C)CO', 'cyclohexane': 'C1CCCCC1',
                  'triethyl phosphate': 'CCO[P(OCC)OCC', 'DMF': 'CN(C)C=O', 'methyl acetate': 'COC(C)=O',
                  'dichloromethane': 'ClCCl', 'tert-butanol': 'CC(C)(C)O', 'acetic acid': 'CC(O)=O',
                  'ethylene glycol': 'OCCO', 'acetonitrile': 'CC#N', 'DMSO': 'CS=O', '2-propoxyethanol': 'CCCOCCO',
                  'n-dodecane': 'CCCCCCCCCCCC', 'formamide': 'NC=O', 'THF': 'C1CCOC1', 'methanol': 'CO',
                  'isopropanol': 'CC(C)O', 'formic acid': 'OC=O', 'water': 'O', 'propionic acid': 'CCC(O)=O',
                  'dipropyl ether': 'CCCOCCC', 'diisopropyl ether': 'CC(C)OC(C)C', '2-methoxyethanol': 'COCCO',
                  'ethyl acetate': 'CCOC(C)=O', 'benzene': 'c1ccccc1', 'p-xylene': 'Cc1ccc(C)cc1',
                  '2-octanol': 'CCCCCCC(C)O', 'MTBE': 'COC(C)(C)C', '1,2-dichloroethane': 'ClCCCl',
                  'ethyl benzene': 'CCc1ccccc1', '2-ethoxyethanol': 'CCOCCO', 'n-propyl acetate': 'CCCOC(C)=O',
                  'methyl propionate': 'CCC(=O)OC', 'triethyl orthoformate': 'CCOC(OCC)OCC',
                  '1-propoxy-2-propanol': 'CCCOCC(C)O', 'diacetone alcohol': 'CC(=O)CC(C)(C)O',
                  'benzyl alcohol': 'OCc1ccccc1', '2-butoxyethanol': 'CCCCOCCO', 'propylene carbonate': 'CC1COC(=O)O1',
                  'nonan-1-ol': 'CCCCCCCCCO', 'sec-pentanol': 'CCCC(C)O', '2-isopropoxyethanol': 'CC(C)OCCO',
                  'DMS': 'CS=O', 'diethylene glycol': 'OCCOCCO', 'MEK': 'CCC(C)=O', 'n-pentyl acetate': 'CCCCCOC(C)=O',
                  'cyclohexanone': 'O=C1CCCCC1', 'tert-amyl alcohol': 'CCC(C)(C)O', 'cyclopentanone': 'O=C1CCCC1',
                  'morpholine': 'C1COCCN1', 'm-xylene': 'Cc1cccc(C)c1', '3-pentanone': 'CCC(=O)CC',
                  '2-ethylhexanol': 'CCCCC(CC)CO', 'n-hexane': 'CCCCCC', 'isooctane': 'CCCCCC(C)C',
                  'isobutanol': 'CC(C)CO', 'isopentanol': 'CC(C)CCO', 'n-hexadecane': 'CCCCCCCCCCCCCCCC',
                  'diethyl ether': 'CCOCC', 'ethylbenzene': 'CCc1ccccc1', 'n-butyric acid': 'CCCC(O)=O',
                  'acrylic acid': 'OC(=O)C=C', '2-ethyl-n-hexanol': 'CCCCC(CC)CO', 'chlorobenzene': 'Clc1ccccc1',
                  '3-oxa-1,5-pentanediol': 'OCCOCCO', '1,3-propanediol': 'OCCCO', 'DMA': 'CC(C)=CCOP(=O)OP(O)=O',
                  'n-dodecanol': 'CCCCCCCCCCCCO', 'n-butyl acetate': 'CCCCOC(C)=O',
                  'methyl 4-tert-butylbenzoate': 'COC(=O)c1ccc(cc1)C(C)(C)C', 'isobutyl acetate': 'CC(C)COC(C)=O',
                  'o-xylene': 'Cc1ccccc1C', '1-methoxy-2-propanol': 'COCC(C)O',
                  '1,2,4-trichlorobenzene': 'Clc1ccc(Cl)c(Cl)c1', 'toluene': 'Cc1ccccc1', 'methyl formate': 'COC=O',
                  'tert-butyl acetate': 'CC(=O)OC(C)(C)C', 'n-octanol': 'CCCCCCCCO', 'propylene glycol': 'CC(O)CO',
                  '4-methylpyridine': 'Cc1ccncc1', 'acetophenone': 'CC(=O)c1ccccc1', '2-pentanone': 'CCCC(C)=O',
                  '1-bromopropane': 'CCCBr', 'p-cymene': 'CC(C)c1ccc(C)cc1', 'cumene': 'CC(C)c1ccccc1',
                  '2-(2-butoxyethoxy)ethanol': 'CCCCOCCOCCO', '1,2-diethoxyethane': 'CCOCCOCC'}

    solvent_mol = Chem.MolFromSmiles(names_dict[str])
    if solvent_mol is not None:
        vector = list(solvent_data[str].loc[use_exp_values])
        if use_maccskeys:
            vector.extend(list(AllChem.GetMACCSKeysFingerprint(solvent_mol)))
        return vector
    else:
        raise ValueError(f"invalid molecule input type: {mol.GetProp('Solvent')}")

def featurize_sdf_with_solvent_experimental_united(path_to_sdf=None, molecules=None, mol_featurizer=ConvMolFeaturizer(),
                               seed=42, shuffle=True, include_descriptors='', test_solvents='', n_test_in_train=0, use_maccskeys=False):
    """"
    Extract molecules from .sdf file and featurize them

    Parameters
    ----------
    path_to_sdf : str
        path to .sdf file with data
        single molecule in .sdf file can contain properties like "logK_{metal}"
        each of these properties will be transformed into a different training sample
    molecules: List[Chem.Mol]
        list of molecules, can be used instead of path_to_sdf
    mol_featurizer : featurizer, optional
        instance of the class used for extracting features of organic molecule
    seed: int = 42
        random seed for reproducibility
    shuffle: bool
        whether to shuffle data after featurization
    include_descriptors: str
        string with additional descriptors separated by a space
    test_solvents: str
        string with solvents to be put in test part (separated by a space)
    n_test_in_train: int
        a number of test-solvent samples in train
    use_maccskeys: boolean
        to use MACCSKeys as a descriptor or not to use

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
    mol_features = [mol_featurizer.featurize(m) for m in mols]

    all_data_train, all_data_test = [], []
    temp_test = []

    for mol_ind in range(len(mols)):
        logS_list = []
        for target in [prop for prop in mols[mol_ind].GetPropNames() if prop.startswith("LogS")]:
            logS_list += [float(mols[mol_ind].GetProp(target))]
            element_symbol = mols[mol_ind].GetProp('Solvent')
            vector = get_solvent_vector_experimental(element_symbol, use_maccskeys)

            vector = np.array(vector, dtype=np.float32)

        features = copy.deepcopy(mol_features[mol_ind])
        features.x_fully_connected = torch.tensor(vector).unsqueeze(0)
        features.y = {"Solubility": torch.tensor([logS_list])}
        if mols[mol_ind].GetProp('Solvent') in test_solvents:
            temp_test += [features]
        else:
            all_data_train += [features]

    if n_test_in_train > len(temp_test):
        raise ValueError("n_test_in_train parameter is more than number of test data")

    selected_vectors = random.sample(temp_test, n_test_in_train)
    all_data_train.extend(selected_vectors)
    all_data_test.extend(vector for vector in temp_test if vector not in selected_vectors)

    if shuffle:
        random.Random(seed).shuffle(all_data_test)
        random.Random(seed).shuffle(all_data_train)

    return all_data_train, all_data_test