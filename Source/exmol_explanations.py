from io import BytesIO
from typing import Union

import exmol
import selfies as sf
from PIL import Image
from cairosvg import svg2png
from rdkit import Chem
from rdkit.Chem import Mol
from torch import nn

from Source.models.GCNN_FCNN.featurizers import Complex


class ModelExmol(nn.Module):
    def __init__(self, base_model, metal, charge, temperature, ionic_str):
        super().__init__()
        self.device = base_model.device
        self.model = base_model
        self.metal = metal
        self.charge = charge
        self.temperature = temperature
        self.ionic_str = ionic_str

    def forward(self, smiles, selfies):
        mol = Chem.MolFromSmiles(sf.decoder(selfies))
        complex = Complex(mol, self.metal, self.charge, self.temperature, self.ionic_str)

        return self.model(complex.graph)["logK"].detach().item()


def explain_by_exmol(model, complex: Complex) -> Image.Image:
    model_exmol = ModelExmol(model, metal=complex.metal, charge=complex.valence, temperature=complex.temperature, ionic_str=complex.ionic_str)
    samples = exmol.sample_space(Chem.MolToSmiles(complex.mol), model_exmol, batched=False, use_selfies=True)
    exmol.lime_explain(samples, descriptor_type='ECFP')
    return Image.open(BytesIO(svg2png(bytestring=exmol.plot_descriptors(samples, return_svg=True))))
