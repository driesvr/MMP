"""Molecular standardization following the matched-molecular-series convention.

Pipeline order (mirrors mmpdb / ChEMBL curation):
  1. Sanitize
  2. Remove explicit Hs
  3. Metal disconnection
  4. Functional-group normalization + reionization  (via Cleanup)
  5. Salt / fragment stripping  (keep largest organic fragment)
  6. Charge neutralization
  7. Tautomer canonicalization
  8. Re-canonicalize SMILES
"""

from __future__ import annotations

import logging

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

logger = logging.getLogger(__name__)

# Pre-build reusable objects (thread-safe for read-only use)
_uncharger = rdMolStandardize.Uncharger()
_te = rdMolStandardize.TautomerEnumerator()


def standardize_mol(mol: Chem.Mol) -> Chem.Mol | None:
    """Apply the full standardization pipeline to an RDKit Mol.

    Returns None if any step fails or produces an empty molecule.
    """
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)
        mol = _uncharger.uncharge(mol)
        mol = _te.Canonicalize(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    except Exception:
        logger.debug("Standardization failed for mol", exc_info=True)
        return None
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return mol


def standardize_smiles(smi: str) -> str | None:
    """Standardize a SMILES string.  Returns canonical SMILES or None."""
    mol = Chem.MolFromSmiles(smi)
    mol = standardize_mol(mol)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)
