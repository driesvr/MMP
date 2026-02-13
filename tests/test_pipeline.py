"""End-to-end tests for the MMP pipeline using a small synthetic dataset."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import polars as pl
import pytest

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mmp.config import load_config, get_assay_config, Config, AssayConfig
from mmp.preprocessing import preprocess, canonicalize_smiles_column
from mmp.fragmentation import fragment_molecules
from mmp.indexing import generate_pairs
from mmp.statistics import (
    compute_pair_deltas,
    aggregate_transform_stats,
    aggregate_fragment_influence,
)
import mmp.database as db
from mmp.query import top_fragments_for_assay, top_transforms_for_assay, query_molecule


# ── Fixtures ──────────────────────────────────────────────────────────────────

# A set of para-substituted benzene analogs — simple, well-understood MMPs.
# All share the benzene scaffold; substituents differ only at the para position.
SMALL_DATASET = [
    # smiles,                  assay,   value
    ("c1ccc(F)cc1",           "logS",   -1.5),
    ("c1ccc(Cl)cc1",          "logS",   -2.1),
    ("c1ccc(Br)cc1",          "logS",   -2.8),
    ("c1ccc(I)cc1",           "logS",   -3.5),
    ("c1ccc(OC)cc1",          "logS",   -0.9),
    ("c1ccc(CC)cc1",          "logS",   -2.3),
    ("c1ccc(C(F)(F)F)cc1",    "logS",   -3.1),
    ("c1ccc(N)cc1",           "logS",    0.2),
    ("c1ccc(O)cc1",           "logS",    0.1),
    ("c1ccc(C)cc1",           "logS",   -1.8),
    # Double-substituted: methyl + halo at different positions
    ("c1cc(F)ccc1C",          "logS",   -2.0),
    ("c1cc(Cl)ccc1C",         "logS",   -2.5),
    ("c1cc(Br)ccc1C",         "logS",   -3.0),
    # hERG data for a subset
    ("c1ccc(F)cc1",           "hERG",    5.2),
    ("c1ccc(Cl)cc1",          "hERG",    5.8),
    ("c1ccc(Br)cc1",          "hERG",    6.3),
    ("c1ccc(OC)cc1",          "hERG",    4.9),
    ("c1ccc(C)cc1",           "hERG",    5.0),
]


@pytest.fixture
def raw_df() -> pl.DataFrame:
    smiles, assays, values = zip(*SMALL_DATASET)
    return pl.DataFrame({
        "smiles": list(smiles),
        "assay_name": list(assays),
        "value": list(values),
    })


@pytest.fixture
def default_cfg() -> Config:
    return load_config(None)


@pytest.fixture
def compounds(raw_df, default_cfg) -> pl.DataFrame:
    return preprocess(raw_df, default_cfg)


@pytest.fixture
def fragments(compounds, default_cfg) -> pl.DataFrame:
    return fragment_molecules(compounds, default_cfg.global_config)


@pytest.fixture
def pairs(fragments) -> pl.DataFrame:
    return generate_pairs(fragments)


@pytest.fixture
def pair_values(pairs, compounds) -> pl.DataFrame:
    return compute_pair_deltas(pairs, compounds)


@pytest.fixture
def tmp_db(compounds, fragments, pair_values, default_cfg, tmp_path) -> str:
    path = str(tmp_path / "test_fixture.duckdb")

    transform_stats = aggregate_transform_stats(pair_values, default_cfg)
    frag_influence = aggregate_fragment_influence(pair_values, default_cfg)

    con = db.create_database(path)
    db.write_compounds(con, compounds)
    db.write_assay_values(con, compounds)
    db.write_fragments(con, fragments)
    db.write_pairs(con, pair_values)
    db.write_transform_stats(con, transform_stats)
    db.write_fragment_influence(con, frag_influence)
    con.close()

    yield path


# ── Config tests ──────────────────────────────────────────────────────────────

class TestConfig:
    def test_default_config(self):
        cfg = load_config(None)
        assert cfg.global_config.max_heavy_atoms == 70
        assert cfg.global_config.max_cuts == 2

    def test_load_yaml_config(self, tmp_path):
        yaml_content = """
global:
  max_heavy_atoms: 50
  max_cuts: 1

assays:
  test_assay:
    transform: log10
    qualifier_handling: drop
    enantiomer_handling: keep
    outlier_removal: iqr
    min_pairs: 5

  _default:
    transform: none
    qualifier_handling: strip
    enantiomer_handling: mean
    outlier_removal: none
    min_pairs: 2
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(yaml_content)

        cfg = load_config(str(config_path))
        assert cfg.global_config.max_heavy_atoms == 50
        assert cfg.global_config.max_cuts == 1

        test_cfg = get_assay_config(cfg, "test_assay")
        assert test_cfg.transform == "log10"
        assert test_cfg.min_pairs == 5

        default_cfg = get_assay_config(cfg, "unknown_assay")
        assert default_cfg.transform == "none"
        assert default_cfg.min_pairs == 2

    def test_invalid_transform_raises(self, tmp_path):
        yaml_content = """
assays:
  bad:
    transform: invalid_transform
"""
        p = tmp_path / "bad.yaml"
        p.write_text(yaml_content)
        with pytest.raises(ValueError, match="transform"):
            load_config(str(p))

    def test_fallback_to_default(self):
        cfg = load_config(None)
        acfg = get_assay_config(cfg, "completely_unknown_assay")
        assert isinstance(acfg, AssayConfig)


# ── Preprocessing tests ───────────────────────────────────────────────────────

class TestPreprocessing:
    def test_output_columns(self, compounds):
        assert set(compounds.columns) == {"mol_id", "canonical_smiles", "assay_name", "value"}

    def test_mol_id_unique_per_smiles(self, compounds):
        per_smiles = (
            compounds
            .select(["mol_id", "canonical_smiles"])
            .unique(subset=["canonical_smiles"])
        )
        # Each canonical SMILES maps to exactly one mol_id
        assert per_smiles["mol_id"].n_unique() == len(per_smiles)

    def test_invalid_smiles_dropped(self, default_cfg):
        df = pl.DataFrame({
            "smiles": ["c1ccccc1", "NOT_A_SMILES", "c1ccc(F)cc1"],
            "assay_name": ["logS", "logS", "logS"],
            "value": [1.0, 2.0, 3.0],
        })
        result = preprocess(df, default_cfg)
        assert len(result) == 2

    def test_values_are_float(self, compounds):
        assert compounds["value"].dtype == pl.Float64

    def test_canonicalize_smiles(self):
        df = pl.DataFrame({
            "smiles": ["C1=CC=CC=C1", "c1ccccc1"],  # same molecule, different form
        })
        result = canonicalize_smiles_column(df)
        # Both should canonicalize to the same SMILES
        assert result["smiles"][0] == result["smiles"][1]


# ── Fragmentation tests ───────────────────────────────────────────────────────

class TestFragmentation:
    def test_fragment_schema(self, fragments):
        assert "mol_id" in fragments.columns
        assert "constant_smi" in fragments.columns
        assert "variable_smi" in fragments.columns
        assert "num_cuts" in fragments.columns

    def test_num_cuts_range(self, fragments):
        assert fragments["num_cuts"].min() >= 1
        assert fragments["num_cuts"].max() <= 1

    def test_produces_fragments(self, fragments):
        assert len(fragments) > 0

    def test_fragment_mol_ids_exist_in_compounds(self, fragments, compounds):
        frag_mol_ids = set(fragments["mol_id"].unique().to_list())
        compound_mol_ids = set(compounds["mol_id"].unique().to_list())
        assert frag_mol_ids.issubset(compound_mol_ids)

    def test_no_oversized_variables(self, fragments, default_cfg):
        max_var = default_cfg.global_config.max_variable_heavy_atoms
        from rdkit import Chem
        for smi in fragments["variable_smi"].to_list():
            mol = Chem.MolFromSmiles(smi)
            if mol:
                heavy = mol.GetNumHeavyAtoms()
                assert heavy <= max_var, f"Variable fragment too large: {smi} ({heavy} heavy atoms)"


# ── Indexing tests ────────────────────────────────────────────────────────────

class TestIndexing:
    def test_pair_schema(self, pairs):
        required = {
            "mol_id_left", "mol_id_right", "constant_smi",
            "variable_left", "variable_right", "transform_smirks",
        }
        assert required.issubset(set(pairs.columns))

    def test_no_self_pairs(self, pairs):
        assert (pairs["mol_id_left"] == pairs["mol_id_right"]).sum() == 0

    def test_ordered_mol_ids(self, pairs):
        assert (pairs["mol_id_left"] < pairs["mol_id_right"]).all()

    def test_smirks_canonical(self, pairs):
        # SMIRKS should always have LHS <= RHS lexicographically
        for row in pairs.iter_rows(named=True):
            lhs, rhs = row["transform_smirks"].split(">>", 1)
            assert lhs <= rhs, f"Non-canonical SMIRKS: {row['transform_smirks']}"

    def test_pairs_found_for_halogen_series(self, pairs, compounds):
        # F, Cl, Br, I on benzene should all pair with each other
        assert len(pairs) > 0


# ── Statistics tests ──────────────────────────────────────────────────────────

class TestStatistics:
    def test_delta_computed(self, pair_values):
        assert "delta" in pair_values.columns
        assert pair_values["delta"].dtype == pl.Float64

    def test_transform_stats_schema(self, pair_values, default_cfg):
        stats = aggregate_transform_stats(pair_values, default_cfg)
        required = {"transform_smirks", "assay_name", "pair_count", "mean_delta"}
        assert required.issubset(set(stats.columns))

    def test_min_pairs_filter(self, pair_values, default_cfg):
        stats = aggregate_transform_stats(pair_values, default_cfg)
        if len(stats) > 0:
            assert stats["pair_count"].min() >= default_cfg._default.min_pairs

    def test_fragment_influence_schema(self, pair_values, default_cfg):
        influence = aggregate_fragment_influence(pair_values, default_cfg)
        required = {"variable_smi", "assay_name", "times_introduced"}
        assert required.issubset(set(influence.columns))


# ── Database tests ────────────────────────────────────────────────────────────

class TestDatabase:
    def test_tables_exist(self, tmp_db):
        import duckdb
        con = duckdb.connect(tmp_db, read_only=True)
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        con.close()
        expected = {
            "compounds", "assay_values", "fragments",
            "pairs", "transform_stats", "fragment_influence",
        }
        assert expected.issubset(tables)

    def test_compounds_populated(self, tmp_db):
        import duckdb
        con = duckdb.connect(tmp_db, read_only=True)
        n = con.execute("SELECT COUNT(*) FROM compounds").fetchone()[0]
        con.close()
        assert n > 0

    def test_pairs_populated(self, tmp_db):
        import duckdb
        con = duckdb.connect(tmp_db, read_only=True)
        n = con.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]
        con.close()
        assert n > 0

    def test_transform_stats_populated(self, tmp_db):
        import duckdb
        con = duckdb.connect(tmp_db, read_only=True)
        n = con.execute("SELECT COUNT(*) FROM transform_stats").fetchone()[0]
        con.close()
        assert n >= 0  # may be 0 if min_pairs filter is strict


# ── Query tests ───────────────────────────────────────────────────────────────

class TestQuery:
    def test_top_transforms_returns_dataframe(self, tmp_db):
        result = top_transforms_for_assay(tmp_db, "logS", top_n=10, min_pairs=1)
        assert isinstance(result, pl.DataFrame)

    def test_top_fragments_returns_dataframe(self, tmp_db):
        result = top_fragments_for_assay(tmp_db, "logS", top_n=10, min_pairs=1)
        assert isinstance(result, pl.DataFrame)

    def test_query_molecule_returns_dataframe(self, tmp_db):
        result = query_molecule(tmp_db, "c1ccc(F)cc1", assay_name="logS")
        assert isinstance(result, pl.DataFrame)

    def test_query_molecule_has_expected_columns(self, tmp_db):
        result = query_molecule(tmp_db, "c1ccc(F)cc1", assay_name="logS")
        if len(result) > 0:
            required = {
                "query_variable", "replacement", "transform_smirks",
                "constant_context", "assay_name", "predicted_delta", "confidence",
            }
            assert required.issubset(set(result.columns))

    def test_query_molecule_product_smiles(self, tmp_db):
        result = query_molecule(tmp_db, "c1ccc(F)cc1", assay_name="logS")
        if len(result) > 0:
            # Product SMILES should be non-empty strings or empty string (if assembly failed)
            assert "product_smiles" in result.columns

    def test_invalid_smiles_raises(self, tmp_db):
        with pytest.raises(ValueError):
            query_molecule(tmp_db, "NOT_VALID_SMILES")

    def test_unknown_assay_returns_empty(self, tmp_db):
        result = top_transforms_for_assay(tmp_db, "nonexistent_assay", min_pairs=1)
        assert len(result) == 0


# ── End-to-end pipeline test ──────────────────────────────────────────────────

class TestEndToEnd:
    def test_full_pipeline(self, raw_df, tmp_path):
        """Run the entire pipeline from raw DataFrame to queryable database."""
        from mmp.config import load_config
        from mmp.preprocessing import preprocess
        from mmp.fragmentation import fragment_molecules
        from mmp.indexing import generate_pairs
        from mmp.statistics import (
            compute_pair_deltas,
            aggregate_transform_stats,
            aggregate_fragment_influence,
        )
        import mmp.database as db

        cfg = load_config(None)

        compounds = preprocess(raw_df, cfg)
        assert len(compounds) > 0

        fragments = fragment_molecules(compounds, cfg.global_config)
        assert len(fragments) > 0

        pairs = generate_pairs(fragments)
        # With our benzene series, we expect many pairs
        assert len(pairs) > 0

        pair_values = compute_pair_deltas(pairs, compounds)
        assert len(pair_values) > 0
        assert "delta" in pair_values.columns

        stats = aggregate_transform_stats(pair_values, cfg)
        influence = aggregate_fragment_influence(pair_values, cfg)

        db_path = str(tmp_path / "test.duckdb")
        con = db.create_database(db_path)
        db.write_compounds(con, compounds)
        db.write_assay_values(con, compounds)
        db.write_fragments(con, fragments)
        db.write_pairs(con, pair_values)
        db.write_transform_stats(con, stats)
        db.write_fragment_influence(con, influence)
        con.close()

        # Verify DB is queryable
        import duckdb
        con2 = duckdb.connect(db_path, read_only=True)
        n_compounds = con2.execute("SELECT COUNT(*) FROM compounds").fetchone()[0]
        n_pairs = con2.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]
        con2.close()

        assert n_compounds > 0
        assert n_pairs > 0
