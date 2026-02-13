"""DuckDB schema creation and bulk write from Polars DataFrames."""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import polars as pl

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS compounds (
    mol_id        INTEGER PRIMARY KEY,
    smiles        TEXT NOT NULL,
    inchi         TEXT,
    num_heavy     INTEGER
);

CREATE TABLE IF NOT EXISTS assay_values (
    mol_id        INTEGER REFERENCES compounds(mol_id),
    assay_name    TEXT NOT NULL,
    value         DOUBLE NOT NULL,
    PRIMARY KEY (mol_id, assay_name)
);

CREATE TABLE IF NOT EXISTS fragments (
    fragment_id   INTEGER PRIMARY KEY,
    mol_id        INTEGER REFERENCES compounds(mol_id),
    constant_smi  TEXT NOT NULL,
    variable_smi  TEXT NOT NULL,
    num_cuts      TINYINT,
    attach_env    TEXT
);

CREATE TABLE IF NOT EXISTS pairs (
    pair_id          INTEGER PRIMARY KEY,
    mol_id_left      INTEGER REFERENCES compounds(mol_id),
    mol_id_right     INTEGER REFERENCES compounds(mol_id),
    constant_smi     TEXT,
    variable_left    TEXT,
    variable_right   TEXT,
    transform_smirks TEXT NOT NULL,
    num_cuts         TINYINT,
    assay_name       TEXT,
    delta            DOUBLE
);

CREATE TABLE IF NOT EXISTS transform_stats (
    transform_smirks TEXT NOT NULL,
    assay_name       TEXT NOT NULL,
    pair_count       INTEGER,
    mean_delta       DOUBLE,
    std_delta        DOUBLE,
    median_delta     DOUBLE,
    min_delta        DOUBLE,
    max_delta        DOUBLE,
    q1_delta         DOUBLE,
    q3_delta         DOUBLE,
    PRIMARY KEY (transform_smirks, assay_name)
);

CREATE TABLE IF NOT EXISTS fragment_influence (
    variable_smi             TEXT NOT NULL,
    assay_name               TEXT NOT NULL,
    times_introduced         INTEGER,
    mean_delta_introduced    DOUBLE,
    std_delta_introduced     DOUBLE,
    PRIMARY KEY (variable_smi, assay_name)
);
"""

_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_frag_constant  ON fragments(constant_smi);
CREATE INDEX IF NOT EXISTS idx_frag_mol       ON fragments(mol_id);
CREATE INDEX IF NOT EXISTS idx_pairs_transform ON pairs(transform_smirks);
CREATE INDEX IF NOT EXISTS idx_pairs_assay     ON pairs(assay_name);
CREATE INDEX IF NOT EXISTS idx_ts_assay        ON transform_stats(assay_name);
CREATE INDEX IF NOT EXISTS idx_fi_assay        ON fragment_influence(assay_name);
"""


def create_database(db_path: str) -> duckdb.DuckDBPyConnection:
    """Create (or open) a DuckDB database and apply the schema."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db_path)
    con.execute(_DDL)
    con.execute(_INDEXES)
    return con


def _insert_polars(con: duckdb.DuckDBPyConnection, table: str, df: pl.DataFrame) -> None:
    """Bulk-insert a Polars DataFrame into *table* via DuckDB."""
    if len(df) == 0:
        return
    # DuckDB can reference a Polars DataFrame directly by name
    con.register("_tmp_df", df)
    con.execute(f"INSERT OR REPLACE INTO {table} SELECT * FROM _tmp_df")
    con.unregister("_tmp_df")


def write_compounds(
    con: duckdb.DuckDBPyConnection,
    compounds: pl.DataFrame,
) -> None:
    """Write unique compounds (mol_id, smiles) with optional InChI and heavy atom count."""
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchi

    rows = []
    seen = set()
    for row in compounds.select(["mol_id", "canonical_smiles"]).unique(subset=["mol_id"]).iter_rows():
        mid, smi = row
        if mid in seen:
            continue
        seen.add(mid)
        mol = Chem.MolFromSmiles(smi) if smi else None
        inchi = MolToInchi(mol) if mol else None
        num_heavy = mol.GetNumHeavyAtoms() if mol else None
        rows.append({"mol_id": mid, "smiles": smi, "inchi": inchi, "num_heavy": num_heavy})

    if rows:
        df = pl.DataFrame(rows, schema={
            "mol_id": pl.Int32,
            "smiles": pl.Utf8,
            "inchi": pl.Utf8,
            "num_heavy": pl.Int32,
        })
        _insert_polars(con, "compounds", df)
        logger.info("Wrote %d compounds", len(df))


def write_assay_values(
    con: duckdb.DuckDBPyConnection,
    compounds: pl.DataFrame,
) -> None:
    """Write (mol_id, assay_name, value) rows."""
    av = compounds.select(["mol_id", "assay_name", "value"]).with_columns(
        pl.col("mol_id").cast(pl.Int32),
        pl.col("value").cast(pl.Float64),
    )
    _insert_polars(con, "assay_values", av)
    logger.info("Wrote %d assay_values rows", len(av))


def write_fragments(
    con: duckdb.DuckDBPyConnection,
    fragments: pl.DataFrame,
) -> None:
    """Write fragment records with auto-generated fragment_id."""
    if len(fragments) == 0:
        return
    frag_df = fragments.with_columns([
        pl.arange(0, len(fragments), dtype=pl.Int32).alias("fragment_id"),
        pl.col("mol_id").cast(pl.Int32),
        pl.col("num_cuts").cast(pl.Int8),
    ]).select(["fragment_id", "mol_id", "constant_smi", "variable_smi", "num_cuts", "attach_env"])
    _insert_polars(con, "fragments", frag_df)
    logger.info("Wrote %d fragments", len(frag_df))


def write_pairs(
    con: duckdb.DuckDBPyConnection,
    pair_values: pl.DataFrame,
) -> None:
    """Write pair records with property deltas."""
    if len(pair_values) == 0:
        return

    needed = [
        "mol_id_left", "mol_id_right", "constant_smi",
        "variable_left", "variable_right", "transform_smirks",
        "num_cuts", "assay_name", "delta",
    ]
    available = [c for c in needed if c in pair_values.columns]
    pairs_df = pair_values.select(available).with_columns([
        pl.arange(0, len(pair_values), dtype=pl.Int32).alias("pair_id"),
        pl.col("mol_id_left").cast(pl.Int32),
        pl.col("mol_id_right").cast(pl.Int32),
        pl.col("num_cuts").cast(pl.Int8),
    ])
    # Reorder to match table schema
    schema_cols = [
        "pair_id", "mol_id_left", "mol_id_right", "constant_smi",
        "variable_left", "variable_right", "transform_smirks",
        "num_cuts", "assay_name", "delta",
    ]
    pairs_df = pairs_df.select([c for c in schema_cols if c in pairs_df.columns])
    _insert_polars(con, "pairs", pairs_df)
    logger.info("Wrote %d pairs", len(pairs_df))


def write_transform_stats(
    con: duckdb.DuckDBPyConnection,
    stats: pl.DataFrame,
) -> None:
    """Write pre-aggregated transform statistics."""
    if len(stats) == 0:
        return
    ts = stats.with_columns(pl.col("pair_count").cast(pl.Int32))
    _insert_polars(con, "transform_stats", ts)
    logger.info("Wrote %d transform_stats rows", len(ts))


def write_fragment_influence(
    con: duckdb.DuckDBPyConnection,
    influence: pl.DataFrame,
) -> None:
    """Write pre-aggregated fragment influence scores."""
    if len(influence) == 0:
        return
    fi = influence.with_columns(pl.col("times_introduced").cast(pl.Int32))
    _insert_polars(con, "fragment_influence", fi)
    logger.info("Wrote %d fragment_influence rows", len(fi))
