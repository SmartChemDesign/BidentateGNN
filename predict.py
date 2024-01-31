from rdkit.Chem import MolFromSmiles as smi2mol  # type: ignore
from rdkit.Chem import MolFromSmarts  # type: ignore
from rdkit.Chem import MolToSmiles as mol2smi  # type: ignore
from rdkit.Chem import rdchem, MACCSkeys, AllChem  # type: ignore
from rdkit.Chem.Draw import MolToImage as mol2img, DrawMorganBit  # type: ignore
from Source.GCNN_FCNN.featurizers import Complex
from Source.GCNN_FCNN.model import GCNN_FCNN
from Source.trainer import ModelShell
from config import ROOT_DIR


# init model object with path to model output
MODEL = ModelShell(GCNN_FCNN, str(ROOT_DIR / "Models" / "5fold_regression_final_model"))

# specify parameters of metal-organic complex
charge = "3"
temperature = "25"
ionic_str = "0.3"
metal = "Am"
molecule = "c1ccccc1"

complex = Complex(mol=molecule, metal=metal,
                  valence=int(charge), temperature=float(temperature), ionic_str=float(ionic_str))

print(MODEL(complex.graph)["logK"].item())
