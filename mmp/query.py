"""Query API: top fragments/transforms for an assay, and per-molecule predictions."""

from __future__ import annotations

import json
import logging

import duckdb
import polars as pl

logger = logging.getLogger(__name__)


# ── Query Mode 1 ─────────────────────────────────────────────────────────────

def top_fragments_for_assay(
    db_path: str,
    assay_name: str,
    top_n: int = 50,
    min_pairs: int = 3,
    sort_by: str = "mean_delta_introduced",
) -> pl.DataFrame:
    """Return the fragments most associated with property change in *assay_name*.

    Parameters
    ----------
    sort_by:
        ``"mean_delta_introduced"`` or ``"times_introduced"``.
    """
    valid_sort = {"mean_delta_introduced", "times_introduced"}
    if sort_by not in valid_sort:
        raise ValueError(f"sort_by must be one of {valid_sort}")

    con = duckdb.connect(db_path, read_only=True)
    try:
        sql = f"""
            SELECT variable_smi, assay_name, times_introduced,
                   mean_delta_introduced, std_delta_introduced
            FROM fragment_influence
            WHERE assay_name = ?
              AND times_introduced >= ?
            ORDER BY {sort_by} DESC
            LIMIT ?
        """
        result = con.execute(sql, [assay_name, min_pairs, top_n]).pl()
    finally:
        con.close()
    return result


def top_transforms_for_assay(
    db_path: str,
    assay_name: str,
    top_n: int = 50,
    min_pairs: int = 3,
    sort_by: str = "mean_delta",
    num_cuts: int | None = None,
) -> pl.DataFrame:
    """Return the transforms most associated with property change in *assay_name*."""
    con = duckdb.connect(db_path, read_only=True)
    try:
        params = [assay_name, min_pairs]
        cuts_clause = ""
        if num_cuts is not None:
            cuts_clause = "AND num_cuts = ?"
            params.append(num_cuts)

        sql = f"""
            SELECT ts.transform_smirks, ts.assay_name, ts.pair_count,
                   ts.mean_delta, ts.std_delta, ts.median_delta,
                   ts.min_delta, ts.max_delta
            FROM transform_stats ts
            WHERE ts.assay_name = ?
              AND ts.pair_count >= ?
              {cuts_clause}
            ORDER BY ABS({sort_by}) DESC
            LIMIT ?
        """
        params.append(top_n)
        result = con.execute(sql, params).pl()
    finally:
        con.close()
    return result


# ── Query Mode 2 ─────────────────────────────────────────────────────────────

def query_molecule(
    db_path: str,
    smiles: str,
    assay_name: str | None = None,
    max_results: int = 100,
    max_variable_heavy_atoms: int = 13,
    min_constant_heavy_atoms: int = 5,
    variable_to_constant_ratio: float = 0.33,
) -> pl.DataFrame:
    """Predict property changes for a query molecule by looking up applicable transforms.

    Looks up transforms directly from ``transform_stats`` by variable fragment,
    so results reflect statistics aggregated across *all* scaffolds — not just
    those where the query's exact constant scaffold appears in the database.

    Parameters
    ----------
    smiles:
        Query molecule SMILES.
    assay_name:
        If given, filter results to this assay only.
    max_results:
        Maximum rows returned.

    Returns
    -------
    DataFrame with columns:
        query_variable, replacement, transform_smirks, constant_context,
        product_smiles, assay_name, predicted_delta, confidence, attach_atom_env
    """
    from rdkit import Chem
    from mmp.fragmentation import _fragment_batch
    from mmp.standardization import standardize_mol

    # ── Standardize and canonicalize query molecule ──────────────────────────
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid query SMILES: {smiles!r}")
    mol = standardize_mol(mol)
    if mol is None:
        raise ValueError(f"Standardization failed for query SMILES: {smiles!r}")
    canon_smi = Chem.MolToSmiles(mol)

    # Assign a temporary mol_id = 0 for the query
    frags = _fragment_batch(
        [(0, canon_smi)],
        max_variable_heavy=max_variable_heavy_atoms,
        min_constant_heavy=min_constant_heavy_atoms,
        variable_ratio=variable_to_constant_ratio,
    )

    if not frags:
        logger.info("No fragments generated for query molecule")
        return pl.DataFrame()

    # ── Look up transforms by variable fragment ──────────────────────────────
    con = duckdb.connect(db_path, read_only=True)
    try:
        rows = []
        seen = set()  # (q_var, replacement, assay_name) dedup key

        assay_clause = "AND ts.assay_name = ?" if assay_name else ""

        for frag in frags:
            q_const = frag["constant_smi"]
            q_var = frag["variable_smi"]
            attach_env = frag.get("attach_env", "")

            # Extract attachment atom type (e.g. "C_SP2") for filtering
            query_atom_type = _parse_attach_atom_type(attach_env)

            # Filter: only propose replacements observed at the same atom type.
            # The replacement is the OTHER side of the transform (not q_var).
            attach_filter = ""
            if query_atom_type:
                attach_filter = """
                    AND EXISTS (
                        SELECT 1 FROM fragments f
                        WHERE f.variable_smi = CASE
                                WHEN split_part(ts.transform_smirks, '>>', 1) = ?
                                THEN split_part(ts.transform_smirks, '>>', 2)
                                ELSE split_part(ts.transform_smirks, '>>', 1)
                              END
                          AND f.attach_env LIKE ?
                    )
                """

            # Find all transforms where q_var appears on either side
            sql = f"""
                SELECT ts.transform_smirks, ts.assay_name,
                       ts.mean_delta, ts.pair_count,
                       ts.std_delta, ts.median_delta,
                       ts.min_delta, ts.max_delta,
                       ts.q1_delta, ts.q3_delta
                FROM transform_stats ts
                WHERE (split_part(ts.transform_smirks, '>>', 1) = ?
                    OR split_part(ts.transform_smirks, '>>', 2) = ?)
                  {assay_clause}
                  {attach_filter}
            """
            params: list = [q_var, q_var]
            if assay_name:
                params.append(assay_name)
            if query_atom_type:
                params.extend([q_var, f"%{query_atom_type}%"])

            stat_rows = con.execute(sql, params).fetchall()

            for (smirks, aname, mean_delta, pair_count,
                 std_delta, median_delta, min_delta, max_delta,
                 q1_delta, q3_delta) in stat_rows:
                lhs, rhs = smirks.split(">>", 1)

                if lhs == q_var:
                    other_var = rhs
                    delta_sign = 1.0   # forward: q_var → other_var
                else:
                    other_var = lhs
                    delta_sign = -1.0  # reverse: other_var → q_var stored, we go opposite

                # Deduplicate across fragments that share the same variable
                dedup_key = (q_var, other_var, aname)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                product = _assemble_product(q_const, other_var)

                # Use median as primary; fall back to mean only if median is absent
                primary_delta = median_delta if median_delta is not None else mean_delta

                # When reversing direction (delta_sign == -1), negate all stats
                # and swap min↔max, q1↔q3 so intervals remain [low, high]
                if delta_sign > 0:
                    row_min = min_delta
                    row_max = max_delta
                    row_q1 = q1_delta
                    row_q3 = q3_delta
                else:
                    row_min = -max_delta if max_delta is not None else None
                    row_max = -min_delta if min_delta is not None else None
                    row_q1 = -q3_delta if q3_delta is not None else None
                    row_q3 = -q1_delta if q1_delta is not None else None

                rows.append({
                    "query_variable": q_var,
                    "replacement": other_var,
                    "transform_smirks": smirks,
                    "constant_context": q_const,
                    "product_smiles": product or "",
                    "assay_name": aname,
                    "predicted_delta": primary_delta * delta_sign,
                    "confidence": pair_count,
                    "attach_atom_env": attach_env,
                    "std_delta": std_delta,
                    "median_delta": median_delta * delta_sign if median_delta is not None else None,
                    "min_delta": row_min,
                    "max_delta": row_max,
                    "q1_delta": row_q1,
                    "q3_delta": row_q3,
                })
    finally:
        con.close()

    if not rows:
        return pl.DataFrame()

    result = pl.DataFrame(rows)
    result = (
        result
        .sort(["confidence", "predicted_delta"], descending=[True, True])
        .head(max_results)
    )
    return result


def _parse_attach_atom_type(attach_env: str) -> str:
    """Extract the atom type (e.g. 'C_SP2') from an attach_env JSON string."""
    if not attach_env:
        return ""
    try:
        env_list = json.loads(attach_env)
        if env_list and isinstance(env_list, list):
            # Format: "1:C_SP2" → "C_SP2"
            entry = env_list[0]
            if ":" in entry:
                return entry.split(":", 1)[1]
    except (json.JSONDecodeError, IndexError, TypeError):
        pass
    return ""


def _assemble_product(constant_smi: str, variable_smi: str) -> str | None:
    """Reconnect constant and variable fragments using molzip."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    try:
        const_mol = Chem.MolFromSmiles(constant_smi)
        var_mol = Chem.MolFromSmiles(variable_smi)
        if const_mol is None or var_mol is None:
            return None

        # molzip: join fragments at matching dummy atom map numbers
        combined = Chem.CombineMols(const_mol, var_mol)
        try:
            product = Chem.molzip(combined)
        except Exception:
            # Fallback: try RWMol-based attachment
            product = _assemble_product_fallback(const_mol, var_mol)
            if product is None:
                return None

        Chem.SanitizeMol(product)
        return Chem.MolToSmiles(product)
    except Exception as exc:
        logger.debug("Product assembly failed: %s", exc)
        return None


def _assemble_product_fallback(const_mol, var_mol) -> "Chem.Mol | None":
    """Fallback product assembly via explicit dummy-atom matching."""
    from rdkit import Chem
    from rdkit.Chem import RWMol

    try:
        combined = Chem.CombineMols(const_mol, var_mol)
        rw = RWMol(combined)

        # Find dummy atoms grouped by atom map number
        dummies: dict[int, list[int]] = {}
        for atom in rw.GetAtoms():
            if atom.GetAtomicNum() == 0:
                map_num = atom.GetAtomMapNum()
                dummies.setdefault(map_num, []).append(atom.GetIdx())

        bonds_to_add: list[tuple[int, int]] = []
        atoms_to_remove: list[int] = []

        for map_num, idxs in dummies.items():
            if len(idxs) != 2:
                return None
            a, b = idxs
            # Get neighbors of each dummy
            nbrs_a = [n.GetIdx() for n in rw.GetAtomWithIdx(a).GetNeighbors()]
            nbrs_b = [n.GetIdx() for n in rw.GetAtomWithIdx(b).GetNeighbors()]
            if len(nbrs_a) != 1 or len(nbrs_b) != 1:
                return None
            bonds_to_add.append((nbrs_a[0], nbrs_b[0]))
            atoms_to_remove.extend([a, b])

        for na, nb in bonds_to_add:
            rw.AddBond(na, nb, Chem.BondType.SINGLE)

        for idx in sorted(atoms_to_remove, reverse=True):
            rw.RemoveAtom(idx)

        Chem.SanitizeMol(rw)
        return rw.GetMol()
    except Exception:
        return None
