"""FastAPI backend for the MMP Explorer web UI."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from mmp.query import query_molecule

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="MMP Explorer")


# ── Pydantic models ──────────────────────────────────────────────────────────

class AssaySelection(BaseModel):
    name: str
    direction: str  # "increase" or "decrease"


class QueryRequest(BaseModel):
    db_path: str
    smiles: str = Field(..., max_length=500)
    assays: list[AssaySelection]
    max_results_per_assay: int = 100


class RenderRequest(BaseModel):
    smiles: str = Field(..., max_length=500)
    width: int = 250
    height: int = 200


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_db(request: Request, db_name: str) -> str:
    """Resolve a database name to a full path, preventing path traversal."""
    db_dir: str = request.app.state.db_dir
    # Only allow simple filenames
    if os.sep in db_name or "/" in db_name or "\\" in db_name or ".." in db_name:
        raise HTTPException(400, "Invalid database name")
    full = os.path.join(db_dir, db_name)
    # Ensure it stays within db_dir
    if not os.path.abspath(full).startswith(os.path.abspath(db_dir)):
        raise HTTPException(400, "Invalid database path")
    if not os.path.isfile(full):
        raise HTTPException(404, f"Database not found: {db_name}")
    return full


def _fragment_query_with_vectors(smiles: str) -> dict[str, list[tuple[str, str]]]:
    """Fragment a molecule and return {vector_key: [(constant_smi, variable_smi), ...]}

    Vector keys encode which bond was cut: "a_b" where a < b (atom indices).
    Uses MMS SMARTS patterns for bond selection (single-cut only).
    """
    from rdkit import Chem
    from mmp.fragmentation import _get_bond_smarts, _normalize_dummies

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(400, f"Invalid SMILES: {smiles!r}")

    total_heavy = mol.GetNumHeavyAtoms()
    bond_smarts = _get_bond_smarts()

    # Collect all cuttable bonds via substructure matching
    cuttable_bonds: dict[int, tuple[int, int]] = {}  # bond_idx -> (atom_a, atom_b)
    for pattern in bond_smarts:
        for match in mol.GetSubstructMatches(pattern):
            a, b = match[0], match[1]
            bond = mol.GetBondBetweenAtoms(a, b)
            if bond is not None:
                cuttable_bonds[bond.GetIdx()] = (a, b)

    vectors: dict[str, list[tuple[str, str]]] = {}

    for bond_idx, (atom_a, atom_b) in cuttable_bonds.items():
        # Vector key from original atom indices (sorted)
        a, b = sorted((atom_a, atom_b))
        vector_key = f"{a}_{b}"

        try:
            frag_mol = Chem.FragmentOnBonds(mol, [bond_idx], dummyLabels=[(1, 1)])
        except Exception:
            continue

        pieces = Chem.GetMolFrags(frag_mol, asMols=True)
        if len(pieces) != 2:
            continue

        # Assign larger = constant, smaller = variable
        heavies = [p.GetNumHeavyAtoms() for p in pieces]
        if heavies[0] >= heavies[1]:
            core_piece, var_piece = pieces[0], pieces[1]
        else:
            core_piece, var_piece = pieces[1], pieces[0]

        # Size filters
        var_heavy = var_piece.GetNumHeavyAtoms()
        const_heavy = core_piece.GetNumHeavyAtoms()
        if var_heavy > 13 or const_heavy < 5:
            continue
        if total_heavy > 0 and var_heavy / total_heavy > 0.33:
            continue

        # Normalize dummies: isotope labels → atom map numbers
        _normalize_dummies(core_piece)
        _normalize_dummies(var_piece)

        try:
            clean_const = Chem.MolToSmiles(core_piece)
            clean_var = Chem.MolToSmiles(var_piece)
        except Exception:
            continue

        if not clean_const or not clean_var:
            continue

        vectors.setdefault(vector_key, []).append((clean_const, clean_var))

    return vectors


def _apply_dark_draw_opts(drawer) -> None:
    """Configure an RDKit SVG drawer for dark backgrounds."""
    opts = drawer.drawOptions()
    opts.setBackgroundColour((0, 0, 0, 0))  # transparent
    opts.updateAtomPalette({
        0: (0.65, 0.65, 0.75),   # dummy/wildcard
        1: (0.82, 0.82, 0.88),   # hydrogen
        6: (0.82, 0.82, 0.88),   # carbon
        7: (0.50, 0.65, 1.0),    # nitrogen
        8: (1.0, 0.42, 0.42),    # oxygen
        9: (0.50, 0.85, 0.50),   # fluorine
        15: (1.0, 0.52, 0.18),   # phosphorus
        16: (0.85, 0.75, 0.20),  # sulfur
        17: (0.42, 0.85, 0.42),  # chlorine
        35: (0.75, 0.42, 0.32),  # bromine
        53: (0.58, 0.32, 0.58),  # iodine
    })


def _render_mol_svg(smiles: str, vectors: dict | None = None,
                    width: int = 500, height: int = 400) -> tuple[str, dict[str, dict]]:
    """Render a molecule to SVG with highlighted vector bonds.

    Returns (svg_string, {vector_key: {x, y, atoms, num_cuts}}).
    """
    from rdkit import Chem, Geometry
    from rdkit.Chem import AllChem, Draw

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "", {}

    AllChem.Compute2DCoords(mol)

    highlight_atoms = []
    highlight_bonds = []
    atom_colors = {}
    bond_colors = {}

    COLOR_BOND = (0.35, 0.58, 0.95)   # bright blue for dark bg

    coord_map: dict[str, dict] = {}

    if vectors:
        for vkey in vectors:
            a, b = [int(x) for x in vkey.split("_")]

            if a < mol.GetNumAtoms() and b < mol.GetNumAtoms():
                highlight_atoms.extend([a, b])
                atom_colors[a] = COLOR_BOND
                atom_colors[b] = COLOR_BOND
                bond_idx = mol.GetBondBetweenAtoms(a, b)
                if bond_idx is not None:
                    highlight_bonds.append(bond_idx.GetIdx())
                    bond_colors[bond_idx.GetIdx()] = COLOR_BOND

            # Compute midpoint of the two atoms for overlay positioning
            conf = mol.GetConformer()
            xs, ys = [], []
            for aidx in (a, b):
                if aidx < mol.GetNumAtoms():
                    pos = conf.GetAtomPosition(aidx)
                    xs.append(pos.x)
                    ys.append(pos.y)

            if xs:
                sym_a = mol.GetAtomWithIdx(a).GetSymbol() if a < mol.GetNumAtoms() else "?"
                sym_b = mol.GetAtomWithIdx(b).GetSymbol() if b < mol.GetNumAtoms() else "?"

                coord_map[vkey] = {
                    "mol_x": sum(xs) / len(xs),
                    "mol_y": sum(ys) / len(ys),
                    "atoms": [a, b],
                    "num_cuts": 1,
                    "atom_symbols": [(sym_a, sym_b)],
                }

    drawer = Draw.MolDraw2DSVG(width, height)
    _apply_dark_draw_opts(drawer)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2
    opts.padding = 0.15

    if highlight_atoms:
        drawer.DrawMolecule(
            mol,
            highlightAtoms=highlight_atoms,
            highlightBonds=highlight_bonds,
            highlightAtomColors=atom_colors,
            highlightBondColors=bond_colors,
        )
    else:
        drawer.DrawMolecule(mol)

    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    # Convert molecule coords → drawing coords for each vector
    for vkey, info in coord_map.items():
        draw_pt = drawer.GetDrawCoords(
            Geometry.Point2D(info["mol_x"], info["mol_y"])
        )
        info["x"] = draw_pt.x
        info["y"] = draw_pt.y
        del info["mol_x"]
        del info["mol_y"]

    return svg, coord_map


def _render_smiles_svg(smiles: str, width: int = 250, height: int = 200) -> str:
    """Render a single SMILES to SVG string."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    AllChem.Compute2DCoords(mol)
    drawer = Draw.MolDraw2DSVG(width, height)
    _apply_dark_draw_opts(drawer)
    drawer.drawOptions().bondLineWidth = 2
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(500, "index.html not found")
    return HTMLResponse(index_path.read_text())


@app.get("/api/databases")
async def list_databases(request: Request):
    db_dir = request.app.state.db_dir
    results = []
    for fname in sorted(os.listdir(db_dir)):
        if fname.endswith(".duckdb"):
            full = os.path.join(db_dir, fname)
            try:
                con = duckdb.connect(full, read_only=True)
                assays = [
                    row[0]
                    for row in con.execute(
                        "SELECT DISTINCT assay_name FROM assay_values ORDER BY assay_name"
                    ).fetchall()
                ]
                con.close()
                results.append({"name": fname, "assays": assays})
            except Exception as exc:
                logger.warning("Skipping %s: %s", fname, exc)
    return results


@app.get("/api/assays")
async def list_assays(request: Request, db_path: str):
    full = _resolve_db(request, db_path)
    con = duckdb.connect(full, read_only=True)
    try:
        rows = con.execute(
            "SELECT DISTINCT assay_name FROM assay_values ORDER BY assay_name"
        ).fetchall()
    finally:
        con.close()
    return [row[0] for row in rows]


@app.post("/api/query")
async def run_query(request: Request, body: QueryRequest):
    db_full = _resolve_db(request, body.db_path)
    canon_smi = _canonicalize(body.smiles)

    # Step 1: Fragment with vector tracking
    vectors = _fragment_query_with_vectors(canon_smi)

    # Step 2: Render SVG
    mol_svg, coord_map = _render_mol_svg(canon_smi, vectors)

    # Step 3: Multi-assay query via existing query_molecule
    try:
        all_results = query_molecule(
            db_full, canon_smi, assay_name=None, max_results=5000
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    if all_results.is_empty():
        return _build_response(canon_smi, mol_svg, coord_map, vectors, {})

    # Filter to selected assays
    assay_names = {a.name for a in body.assays}
    direction_map = {a.name: a.direction for a in body.assays}
    filtered = all_results.filter(pl.col("assay_name").is_in(list(assay_names)))

    if filtered.is_empty():
        return _build_response(canon_smi, mol_svg, coord_map, vectors, {})

    # Step 4: Group proposals and map to vectors
    proposals_by_vector = _map_proposals_to_vectors(
        filtered, vectors, direction_map, body.max_results_per_assay
    )

    return _build_response(canon_smi, mol_svg, coord_map, vectors, proposals_by_vector)


def _canonicalize(smiles: str) -> str:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(400, f"Invalid SMILES: {smiles!r}")
    return Chem.MolToSmiles(mol)


def _map_proposals_to_vectors(
    results: pl.DataFrame,
    vectors: dict[str, list[tuple[str, str]]],
    direction_map: dict[str, str],
    max_per_assay: int,
) -> dict[str, list[dict]]:
    """Match query results to vectors and build the proposals_by_vector structure."""

    # Build a lookup: (constant_smi, variable_smi) -> vector_key
    frag_to_vector: dict[tuple[str, str], str] = {}
    for vkey, frag_list in vectors.items():
        for const_smi, var_smi in frag_list:
            frag_to_vector[(const_smi, var_smi)] = vkey

    # Group results by (constant_context, query_variable, replacement, product_smiles)
    grouped: dict[tuple[str, str, str, str], list[dict]] = {}
    for row in results.iter_rows(named=True):
        key = (
            row["constant_context"],
            row["query_variable"],
            row["replacement"],
            row["product_smiles"],
        )
        grouped.setdefault(key, []).append(row)

    proposals_by_vector: dict[str, list[dict]] = {}

    for (const_ctx, q_var, replacement, product_smi), assay_rows in grouped.items():
        # Find which vector this maps to
        vkey = frag_to_vector.get((const_ctx, q_var))
        if vkey is None:
            continue

        # Build per-assay deltas
        assay_deltas = []
        net_score = 0.0
        for row in assay_rows:
            aname = row["assay_name"]
            if aname not in direction_map:
                continue
            delta = row["predicted_delta"]
            confidence = row["confidence"]
            direction = direction_map[aname]
            direction_sign = 1.0 if direction == "increase" else -1.0
            is_favorable = (delta > 0) == (direction == "increase")
            assay_deltas.append({
                "assay_name": aname,
                "predicted_delta": round(delta, 4),
                "confidence": confidence,
                "direction": direction,
                "is_favorable": is_favorable,
                "std_delta": round(row.get("std_delta") or 0.0, 4),
                "median_delta": round(row.get("median_delta") or 0.0, 4),
                "min_delta": round(row.get("min_delta") or 0.0, 4),
                "max_delta": round(row.get("max_delta") or 0.0, 4),
                "q1_delta": round(row.get("q1_delta") or 0.0, 4),
                "q3_delta": round(row.get("q3_delta") or 0.0, 4),
            })
            weight = math.log10(confidence + 1) if confidence > 0 else 0.1
            net_score += delta * direction_sign * weight

        if not assay_deltas:
            continue

        proposal = {
            "replacement": replacement,
            "product_smiles": product_smi,
            "constant_context": const_ctx,
            "assay_deltas": assay_deltas,
            "net_score": round(net_score, 4),
        }

        proposals_by_vector.setdefault(vkey, []).append(proposal)

    # Sort each vector's proposals by net_score descending
    for vkey in proposals_by_vector:
        proposals_by_vector[vkey].sort(key=lambda p: p["net_score"], reverse=True)

    return proposals_by_vector


def _build_response(
    canon_smi: str,
    mol_svg: str,
    coord_map: dict[str, dict],
    vectors: dict[str, list],
    proposals_by_vector: dict[str, list[dict]],
) -> dict[str, Any]:
    """Assemble the full JSON response."""
    vector_list = []
    for vkey, info in coord_map.items():
        a, b = [int(x) for x in vkey.split("_")]

        atom_symbols_flat = []
        for syms in info.get("atom_symbols", []):
            atom_symbols_flat.extend(syms)

        vector_list.append({
            "vector_id": vkey,
            "bond_atoms": [[a, b]],
            "atom_symbols": atom_symbols_flat,
            "num_cuts": 1,
            "draw_coords": {"x": round(info["x"], 1), "y": round(info["y"], 1)},
            "num_proposals": len(proposals_by_vector.get(vkey, [])),
        })

    return {
        "canonical_smiles": canon_smi,
        "molecule_svg": mol_svg,
        "vectors": vector_list,
        "proposals_by_vector": proposals_by_vector,
    }


@app.post("/api/render_molecule")
async def render_molecule(body: RenderRequest):
    svg = _render_smiles_svg(body.smiles, body.width, body.height)
    if not svg:
        raise HTTPException(400, f"Cannot render SMILES: {body.smiles!r}")
    return {"svg": svg}
