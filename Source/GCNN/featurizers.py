#import dgl
import numpy as np
import pandas as pd
import torch
#from dgllife.utils import mol_to_bigraph
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from collections import defaultdict
import random

from Source.GCNN_FCNN.featurizers import convert_to_float32

class VecData(Dataset):
    def __init__(self, x=None, y=None):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def one_of_k_encoding(x, allowable_set):
    """Encodes elements of a provided set as integers.

      Parameters
      ----------
      x: object
        Must be present in `allowable_set`.
      allowable_set: list
        List of allowable quantities.

      Returns
      -------
      encoding: list
        encoding[i] == 1 if allowable_set[i] == x
        encoding[i] == 0 otherwise
        For example, [True, False, False].

      Raises
      ------
      `ValueError` if `x` is not in `allowable_set`.
    """
    if x not in allowable_set:
        raise ValueError("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element.

      Unlike `one_of_k_encoding`, if `x` is not in `allowable_set`, this method
      pretends that `x` is the last element of `allowable_set`.

      Parameters
      ----------
      x: object
        Must be present in `allowable_set`.
      allowable_set: list
        List of allowable quantities.

      Returns
      --------
      encoding: list
        encoding[i] == 1 if allowable_set[i] == x or x not in allowable_set
        encoding[i] == 0 otherwise
        For example, [True, False, False]. For example, [False, False, True].
      """
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def adj_to_edge(adj_list: list) -> torch.Tensor:
    """
    Convert adjacency list of a graph to list of its edges

    Parameters
    ----------
    adj_list : list
        Adjacency list of a graph, i.e. adj_list[i] is a list of neighbors of i-th vertex

    Returns
    -------
    edge_list : list
       List of graph edges in format [start_vertex, end_vertex]
    """
    edge_list = []
    for i, current_neighbours in enumerate(adj_list):
        for neighbour in current_neighbours:
            edge_list.append([i, neighbour])

    return torch.tensor(edge_list, dtype=torch.int64)


def atom_features(atom,
                  bool_id_feat=False,
                  explicit_H=False,
                  use_chirality=False):
    """
    Helper method used to compute per-atom feature vectors.
    Many different featurization methods compute per-atom features such as ConvMolFeaturizer, WeaveFeaturizer. This method computes such features.

    Parameters
    ----------
    bool_id_feat: bool, optional
    Return an array of unique identifiers corresponding to atom type.
    explicit_H: bool, optional
    If true, model hydrogens explicitly
    use_chirality: bool, optional
    If true, use chirality information.

    Returns
    -------
    np.ndarray of per-atom features.
    """
    if bool_id_feat:
        return  # np.array([atom_to_id(atom)])
    else:
        from rdkit import Chem
        results = one_of_k_encoding_unk(
            atom.GetSymbol(),
            [
                'C',
                'N',
                'O',
                'S',
                'F',
                'Si',
                'P',
                'Cl',
                'Br',
                'Mg',
                'Na',
                'Ca',
                'Fe',
                'As',
                'Al',
                'I',
                'B',
                'V',
                'K',
                'Tl',
                'Yb',
                'Sb',
                'Sn',
                'Ag',
                'Pd',
                'Co',
                'Se',
                'Ti',
                'Zn',
                'H',  # H? H!
                'Li',
                'Ge',
                'Cu',
                'Au',
                'Ni',
                'Cd',
                'In',
                'Mn',
                'Zr',
                'Cr',
                'Pt',
                'Hg',
                'Pb',
                'Unknown'
            ]) + one_of_k_encoding(atom.GetDegree(),
                                   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
                  one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  one_of_k_encoding_unk(atom.GetHybridization(), [
                      Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2
                  ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if not explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                      [0, 1, 2, 3, 4])
        if use_chirality:
            try:
                results = results + one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            except:
                results = results + [False, False
                                     ] + [atom.HasProp('_ChiralityPossible')]

        return np.array(results)


class DGLFeaturizer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def featurize(self, mol):
        dgl_graph = mol_to_bigraph(mol, **self.kwargs)
        networkx_graph = dgl.to_networkx(dgl_graph)
        graph = from_networkx(networkx_graph)
        graph.x = dgl_graph.ndata['h'] if 'h' in dgl_graph.ndata else None
        graph.edge_attr = dgl_graph.edata['e'] if 'e' in dgl_graph.edata else None
        graph.id = None
        return graph


class ConvMolFeaturizer():
    name = ['conv_mol']

    def __init__(self, master_atom=False, use_chirality=False,
                 atom_properties=[]):
        """
        Parameters
        ----------
        master_atom: Boolean
          if true create a fake atom with bonds to every other atom.
          the initialization is the mean of the other atom features in
          the molecule.  This technique is briefly discussed in
          Neural Message Passing for Quantum Chemistry
          https://arxiv.org/pdf/1704.01212.pdf
        use_chirality: Boolean
          if true then make the resulting atom features aware of the
          chirality of the molecules in question
        atom_properties: list of string or None
          properties in the RDKit Mol object to use as additional
          atom-level features in the larger molecular feature.  If None,
          then no atom-level properties are used.  Properties should be in the
          RDKit mol object should be in the form
          atom XXXXXXXX NAME
          where XXXXXXXX is a zero-padded 8 digit number coresponding to the
          zero-indexed atom index of each atom and NAME is the name of the property
          provided in atom_properties.  So "atom 00000000 sasa" would be the
          name of the molecule level property in mol where the solvent
          accessible surface area of atom 0 would be stored.

        Since ConvMol is an object and not a numpy array, need to set dtype to
        object.
        """
        self.dtype = object
        self.master_atom = master_atom
        self.use_chirality = use_chirality
        self.atom_properties = list(atom_properties)

    def _get_atom_properties(self, atom):
        """
        For a given input RDKit atom return the values of the properties
        requested when initializing the featurize.  See the __init__ of the
        class for a full description of the names of the properties

        Parameters
        ----------
        atom: RDKit.rdchem.Atom
          Atom to get the properties of
        returns a numpy lists of floats of the same size as self.atom_properties
        """
        values = []
        for prop in self.atom_properties:
            mol_prop_name = str("atom %08d %s" % (atom.GetIdx(), prop))
            try:
                values.append(float(atom.GetOwningMol().GetProp(mol_prop_name)))
            #     values.extend(fukui_desc)
            except KeyError:
                raise KeyError("No property %s found in %s in %s" %
                               (mol_prop_name, atom.GetOwningMol(), self))
        return np.array(values)

    def get_fukui(self, mol) -> []:
        pass

    def featurize(self, mol, fukui_conf=100):
        """Encodes mol as a ConvMol object."""
        # Get the node features
        idx_nodes = [(a.GetIdx(),
                      np.concatenate((atom_features(
                          a, use_chirality=self.use_chirality),
                                      self._get_atom_properties(a))))
                     for a in mol.GetAtoms()]

        idx_nodes.sort()  # Sort by ind to ensure same order as rd_kit
        idx, nodes = list(zip(*idx_nodes))

        # Stack nodes into an array
        nodes = np.vstack(nodes)
        if self.master_atom:
            master_atom_features = np.expand_dims(np.mean(nodes, axis=0), axis=0)
            nodes = np.concatenate([nodes, master_atom_features], axis=0)

        # Get bond lists with reverse edges included
        edge_list = [
            (b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()
        ]

        # Get canonical adjacency list
        canon_adj_list = [[] for mol_id in range(len(nodes))]
        for edge in edge_list:
            canon_adj_list[edge[0]].append(edge[1])
            canon_adj_list[edge[1]].append(edge[0])

        if self.master_atom:
            fake_atom_index = len(nodes) - 1
            for index in range(len(nodes) - 1):
                canon_adj_list[index].append(fake_atom_index)

        return Data(x=torch.tensor(nodes, dtype=torch.float32),
                    edge_index=adj_to_edge(canon_adj_list).t().contiguous())

    def feature_length(self):
        return 75 + len(self.atom_properties)

    def __hash__(self):
        atom_properties = tuple(self.atom_properties)
        return hash((self.master_atom, self.use_chirality, atom_properties))

    def __eq__(self, other):
        if not isinstance(self, other.__class__):
            return False
        return self.master_atom == other.master_atom and \
            self.use_chirality == other.use_chirality and \
            tuple(self.atom_properties) == tuple(other.atom_properties)



def featurize_df(df: pd.DataFrame, mol_featurizer, target="logK"):
    molecules = df["smiles"].apply(lambda s: Chem.MolFromSmiles(s))
    graphs = molecules.apply(lambda m: mol_featurizer.featurize(m)).tolist()
    targets = df[target].tolist()
    for graph, target_value in zip(graphs, targets):
        graph.y = {target: torch.tensor([[float(target_value)]])}

    return graphs

def featurize_sdf(path_to_sdf: str, mol_featurizer, target="logK"):
    print(type(path_to_sdf))
    molecules = [i for i in Chem.SDMolSupplier(path_to_sdf)]
    graphs = [mol_featurizer.featurize(m) for m in molecules]
    targets = [float(mol.GetProp(target)) for mol in molecules]
    for graph, target_value in zip(graphs, targets):
        graph.y = {target: torch.tensor([[float(target_value)]])}

    return graphs

def featurize_sdf_exp_no_solvent(
    path_to_sdf,
    mol_featurizer,
    output_dataset_path=None,
    seed=42,
    shuffle=True,
):
    """
    Featurize molecules from an SDF file using only experimental redox potentials.
    Duplicate SMILES entries are averaged. Solvent information is ignored.

    Args:
        path_to_sdf (str): Path to the SDF file.
        mol_featurizer: Mol featurizer instance (e.g. ConvMolFeaturizer).
        output_dataset_path (str, optional): Base path for saving CSV outputs.
        seed (int): Random seed for shuffling.
        shuffle (bool): Whether to shuffle output lists.

    Returns:
        tuple: (ox_data, red_data) — lists of torch_geometric Data objects
               for oxidation and reduction, respectively.
    """
    molecules = [m for m in Chem.SDMolSupplier(str(path_to_sdf)) if m is not None]
    mol_features = [mol_featurizer.featurize(m) for m in molecules]

    grouped_data = defaultdict(list)
    dataset_records = []

    for mol_idx, mol in enumerate(molecules):
        mol_props = mol.GetPropNames()

        if "redox_type" not in mol_props:
            continue

        redox_type_full = mol.GetProp("redox_type").strip().lower()
        if redox_type_full == "oxidation":
            redox_type = "ox"
        elif redox_type_full == "reduction":
            redox_type = "red"
        else:
            continue

        E = float(mol.GetProp(f"E_{redox_type}_V"))
        smiles = Chem.MolToSmiles(mol)

        grouped_data[(smiles, redox_type)].append({
            "mol_feature": mol_features[mol_idx],
            "E": E,
        })
        dataset_records.append({
            "molecule_smiles": smiles,
            "redox_type": redox_type,
            "E_value": E,
            "data_type": "exp",
        })

    ox_data = []
    red_data = []
    averaged_records = []

    for (smiles, redox_type), entries in grouped_data.items():
        avg_E = np.mean([e["E"] for e in entries])
        features = entries[0]["mol_feature"].clone()
        features.y = {"E": torch.tensor([[avg_E]])}
        features.smiles = smiles

        (ox_data if redox_type == "ox" else red_data).append(features)
        averaged_records.append({
            "molecule_smiles": smiles,
            "redox_type": redox_type,
            "E_avg": avg_E,
            "data_type": "exp",
        })

    if output_dataset_path:
        pd.DataFrame(averaged_records).to_csv(
            output_dataset_path.replace(".csv", "_averaged.csv"), index=False
        )
        pd.DataFrame(dataset_records).to_csv(
            output_dataset_path.replace(".csv", "_raw.csv"), index=False
        )

    ox_data = convert_to_float32(ox_data)
    red_data = convert_to_float32(red_data)

    if shuffle:
        random.Random(seed).shuffle(ox_data)
        random.Random(seed).shuffle(red_data)

    return ox_data, red_data
