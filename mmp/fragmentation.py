"""MMS-style fragmentation using reaction SMARTS with multiprocessing."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from mmp.config import GlobalConfig

logger = logging.getLogger(__name__)

# Schema for the fragment records DataFrame
FRAGMENT_SCHEMA = {
    "mol_id": pl.UInt32,
    "constant_smi": pl.Utf8,
    "variable_smi": pl.Utf8,
    "num_cuts": pl.UInt8,
    "attach_env": pl.Utf8,
}

# MMS bond-selection SMARTS (single-cut only):
#   1. Exocyclic bonds: ring atom to any atom, non-ring bond
#   2. Heteroatom–sp3-carbon bonds: both non-ring
BOND_SMARTS: list = []  # populated lazily in worker processes

def _get_bond_smarts():
    """Return compiled SMARTS patterns (lazy init for pickling across processes)."""
    from rdkit import Chem
    return [
        Chem.MolFromSmarts('[*;R:1]-!@[*:2]'),
        Chem.MolFromSmarts('[!#6;!R:1]-!@[C;!X3;!R:2]'),
    ]


def _normalize_dummies(mol):
    """Convert isotope-labeled dummies [1*] to atom-map-numbered [*:1]."""
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            iso = atom.GetIsotope()
            if iso:
                atom.SetIsotope(0)
                atom.SetAtomMapNum(iso)
            elif atom.GetAtomMapNum() == 0:
                atom.SetAtomMapNum(1)


# ── Worker function (must be top-level for pickling) ─────────────────────────

def _fragment_batch(
    batch: list[tuple[int, str]],
    max_variable_heavy: int,
    min_constant_heavy: int,
    variable_ratio: float,
) -> list[dict]:
    """Fragment a batch of (mol_id, smiles) tuples. Runs in a worker process."""
    from rdkit import Chem

    bond_smarts = _get_bond_smarts()

    records = []
    for mol_id, smi in batch:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        total_heavy = mol.GetNumHeavyAtoms()
        if total_heavy == 0:
            continue

        # Collect all cuttable bond indices via substructure matching
        cuttable_bonds: set[int] = set()
        for pattern in bond_smarts:
            for match in mol.GetSubstructMatches(pattern):
                a, b = match[0], match[1]
                bond = mol.GetBondBetweenAtoms(a, b)
                if bond is not None:
                    cuttable_bonds.add(bond.GetIdx())

        # Single-cut each cuttable bond
        for bond_idx in cuttable_bonds:
            try:
                frag_mol = Chem.FragmentOnBonds(
                    mol, [bond_idx], dummyLabels=[(1, 1)]
                )
            except Exception:
                continue

            pieces = Chem.GetMolFrags(frag_mol, asMols=True)
            if len(pieces) != 2:
                continue

            # Assign larger = constant, smaller = variable
            heavies = [p.GetNumHeavyAtoms() for p in pieces]
            if heavies[0] >= heavies[1]:
                core_mol, variable_mol = pieces[0], pieces[1]
            else:
                core_mol, variable_mol = pieces[1], pieces[0]

            # Size filters
            var_heavy = variable_mol.GetNumHeavyAtoms()
            const_heavy = core_mol.GetNumHeavyAtoms()

            if var_heavy > max_variable_heavy:
                continue
            if const_heavy < min_constant_heavy:
                continue
            if total_heavy > 0 and var_heavy / total_heavy > variable_ratio:
                continue

            num_cuts = 1

            # Normalize dummy atoms: isotope labels → atom map numbers
            _normalize_dummies(core_mol)
            _normalize_dummies(variable_mol)

            # Attachment environment from constant fragment
            attach_env = _get_attach_env(core_mol)

            try:
                const_smi = Chem.MolToSmiles(core_mol)
                var_smi = Chem.MolToSmiles(variable_mol)
            except Exception:
                continue

            records.append({
                "mol_id": mol_id,
                "constant_smi": const_smi,
                "variable_smi": var_smi,
                "num_cuts": num_cuts,
                "attach_env": attach_env,
            })

    return records


def _get_attach_env(core_mol) -> str:
    """Encode attachment atom environments from dummy atom neighbors in core."""
    import json
    from rdkit import Chem

    envs = []
    for atom in core_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:  # dummy [*:n]
            for neighbor in atom.GetNeighbors():
                sym = neighbor.GetSymbol()
                hyb = str(neighbor.GetHybridization()).split(".")[-1]
                map_num = atom.GetAtomMapNum()
                envs.append(f"{map_num}:{sym}_{hyb}")

    envs.sort()
    return json.dumps(envs)


# ── Public API ────────────────────────────────────────────────────────────────

def fragment_molecules(
    compounds: pl.DataFrame,
    gcfg: "GlobalConfig",
) -> pl.DataFrame:
    """Fragment all unique molecules in *compounds*.

    Parameters
    ----------
    compounds:
        DataFrame with at least ``mol_id`` (uint32) and ``canonical_smiles`` columns.
    gcfg:
        Global configuration.

    Returns
    -------
    Polars DataFrame with columns matching FRAGMENT_SCHEMA.
    """
    # Deduplicate: each unique SMILES fragmented once
    unique = (
        compounds.select(["mol_id", "canonical_smiles"])
        .unique(subset=["canonical_smiles"])
        .sort("mol_id")
    )

    # Pre-filter by heavy atom count
    from rdkit import Chem

    mol_ids = unique["mol_id"].to_list()
    smiles_list = unique["canonical_smiles"].to_list()

    filtered: list[tuple[int, str]] = []
    for mid, smi in zip(mol_ids, smiles_list):
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            logger.warning("Pre-filter: invalid SMILES for mol_id %d", mid)
            continue
        if mol.GetNumHeavyAtoms() > gcfg.max_heavy_atoms:
            logger.debug("Pre-filter: mol_id %d too large (%d heavy atoms)", mid, mol.GetNumHeavyAtoms())
            continue
        filtered.append((mid, smi))

    if not filtered:
        return pl.DataFrame(schema=FRAGMENT_SCHEMA)

    n_workers = gcfg.n_workers if gcfg.n_workers > 0 else os.cpu_count() or 1
    chunk_size = 1000
    chunks = [filtered[i: i + chunk_size] for i in range(0, len(filtered), chunk_size)]

    all_records: list[dict] = []

    if n_workers == 1 or len(chunks) == 1:
        for chunk in chunks:
            recs = _fragment_batch(
                chunk,
                gcfg.max_variable_heavy_atoms,
                gcfg.min_constant_heavy_atoms,
                gcfg.variable_to_constant_ratio,
            )
            all_records.extend(recs)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    _fragment_batch,
                    chunk,
                    gcfg.max_variable_heavy_atoms,
                    gcfg.min_constant_heavy_atoms,
                    gcfg.variable_to_constant_ratio,
                ): i
                for i, chunk in enumerate(chunks)
            }
            for fut in as_completed(futures):
                try:
                    all_records.extend(fut.result())
                except Exception as exc:
                    logger.error("Fragment batch %d failed: %s", futures[fut], exc)

    if not all_records:
        return pl.DataFrame(schema=FRAGMENT_SCHEMA)

    return pl.DataFrame(all_records, schema=FRAGMENT_SCHEMA)
