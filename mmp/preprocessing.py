"""Data preprocessing: qualifier handling, transforms, enantiomers, outliers."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from mmp.config import AssayConfig, Config

logger = logging.getLogger(__name__)

# Matches optional qualifier prefix at the start of a value string, e.g. ">", "<=", "~"
_QUALIFIER_RE = re.compile(r"^\s*([><=!~]+)\s*")


# ── Per-column transforms ────────────────────────────────────────────────────

def _apply_transform(series: pl.Series, transform: str) -> pl.Series:
    if transform == "none":
        return series
    if transform == "log10":
        return series.log(10)
    if transform == "neg_log10":
        return series.log(10).neg()
    if transform == "pIC50":
        # Assumes values are in µM: pIC50 = 6 - log10(value)
        return 6.0 - series.log(10)
    raise ValueError(f"Unknown transform: {transform!r}")


# ── Enantiomer deduplication ─────────────────────────────────────────────────

def _canonical_inchi_no_stereo(smiles_series: pl.Series) -> pl.Series:
    """Return InChI without stereo layer (/t /m /s) for each SMILES."""
    try:
        from rdkit import Chem
        from rdkit.Chem.inchi import MolToInchi
    except ImportError as exc:
        raise ImportError("RDKit is required for enantiomer handling") from exc

    inchis = []
    for smi in smiles_series.to_list():
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            inchis.append(None)
        else:
            inchi = MolToInchi(mol, options="/SNon") or ""
            inchis.append(inchi)
    return pl.Series(smiles_series.name, inchis, dtype=pl.Utf8)


# ── Outlier removal ──────────────────────────────────────────────────────────

def _remove_outliers_iqr(df: pl.DataFrame) -> pl.DataFrame:
    q1 = df["value"].quantile(0.25)
    q3 = df["value"].quantile(0.75)
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    return df.filter((pl.col("value") >= lo) & (pl.col("value") <= hi))


def _remove_outliers_zscore(df: pl.DataFrame) -> pl.DataFrame:
    mean = df["value"].mean()
    std = df["value"].std()
    if std == 0 or std is None:
        return df
    z = ((pl.col("value") - mean) / std).abs()
    return df.filter(z <= 3.0)


# ── SMILES canonicalization ──────────────────────────────────────────────────

def canonicalize_smiles_column(df: pl.DataFrame, smiles_col: str = "smiles") -> pl.DataFrame:
    """Re-canonicalize SMILES via RDKit; drop invalid rows."""
    from rdkit import Chem

    canonical = []
    valid_mask = []
    for smi in df[smiles_col].to_list():
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            logger.warning("Invalid SMILES dropped: %r", smi)
            valid_mask.append(False)
            canonical.append(None)
        else:
            valid_mask.append(True)
            canonical.append(Chem.MolToSmiles(mol))

    df = df.with_columns(pl.Series(smiles_col, canonical, dtype=pl.Utf8))
    df = df.filter(pl.Series("_valid", valid_mask))
    return df


# ── Main entry point ─────────────────────────────────────────────────────────

def _parse_value_and_qualifier(df: pl.DataFrame) -> pl.DataFrame:
    """Parse qualifier prefixes embedded in the value column (e.g. '>5.0' → qualifier='>', value=5.0).

    If a separate 'qualifier' column already exists it is left untouched and only
    rows without a qualifier column entry are examined for embedded prefixes.
    If the value column is already numeric, no parsing is needed.
    """
    # If value column is already numeric, nothing to do
    if df["value"].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt32, pl.UInt64):
        if "qualifier" not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias("qualifier"))
        return df

    # Value column is string — strip qualifier prefixes
    raw_values = df["value"].cast(pl.Utf8).to_list()
    clean_values: list = []
    qualifiers: list = []
    for v in raw_values:
        if v is None:
            clean_values.append(None)
            qualifiers.append(None)
            continue
        m = _QUALIFIER_RE.match(v)
        if m:
            qualifiers.append(m.group(1))
            clean_values.append(v[m.end():].strip())
        else:
            qualifiers.append(None)
            clean_values.append(v)

    df = df.with_columns([
        pl.Series("value", clean_values, dtype=pl.Utf8),
    ])
    # Only write extracted qualifiers if no pre-existing qualifier column
    if "qualifier" not in df.columns:
        df = df.with_columns(pl.Series("qualifier", qualifiers, dtype=pl.Utf8))
    return df


def preprocess(
    df: pl.DataFrame,
    cfg: "Config",
) -> pl.DataFrame:
    """Clean and canonicalize a raw input DataFrame.

    Expected input columns: smiles, assay_name, value
    Optional column: qualifier  (or qualifiers embedded in the value string, e.g. '>5.0')

    Returns a DataFrame with columns: mol_id, canonical_smiles, assay_name, value
    """
    from mmp.config import get_assay_config

    required = {"smiles", "assay_name", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame missing columns: {missing}")

    # Parse embedded qualifiers (e.g. '>5.0') and ensure numeric value column
    df = _parse_value_and_qualifier(df)
    df = df.with_columns(pl.col("value").cast(pl.Float64, strict=False))
    df = df.filter(pl.col("value").is_not_null())

    # ── 1. Canonicalize SMILES ───────────────────────────────────────────────
    df = canonicalize_smiles_column(df, "smiles")
    df = df.rename({"smiles": "canonical_smiles"})

    # ── 2. Per-assay processing ──────────────────────────────────────────────
    assay_frames = []
    for assay_name, group in df.group_by("assay_name"):
        assay_name = assay_name[0] if isinstance(assay_name, (list, tuple)) else assay_name
        acfg: AssayConfig = get_assay_config(cfg, assay_name)
        group = _process_assay(group, assay_name, acfg)
        if group is not None and len(group) > 0:
            assay_frames.append(group)

    if not assay_frames:
        return pl.DataFrame(schema={
            "mol_id": pl.UInt32,
            "canonical_smiles": pl.Utf8,
            "assay_name": pl.Utf8,
            "value": pl.Float64,
        })

    result = pl.concat(assay_frames)

    # ── 3. Assign mol_id ─────────────────────────────────────────────────────
    smiles_list = result["canonical_smiles"].unique().sort().to_list()
    smiles_to_id = {smi: i for i, smi in enumerate(smiles_list)}
    mol_ids = pl.Series(
        "mol_id",
        [smiles_to_id[s] for s in result["canonical_smiles"].to_list()],
        dtype=pl.UInt32,
    )
    result = result.with_columns(mol_ids)

    return result.select(["mol_id", "canonical_smiles", "assay_name", "value"])


def _process_assay(
    df: pl.DataFrame,
    assay_name: str,
    acfg: "AssayConfig",
) -> pl.DataFrame | None:
    # ── Qualifier handling ───────────────────────────────────────────────────
    # The 'qualifier' column is always present after _parse_value_and_qualifier
    # (either from the input CSV or extracted from the value string).
    if acfg.qualifier_handling == "drop":
        df = df.filter(
            pl.col("qualifier").is_null() | (pl.col("qualifier").str.strip_chars() == "")
        )
    # "strip": qualifiers already removed from value string during _parse_value_and_qualifier

    if len(df) == 0:
        return None

    # ── Transform ────────────────────────────────────────────────────────────
    if acfg.transform != "none":
        # Filter non-positive values before log transforms
        df = df.filter(pl.col("value") > 0)
        if len(df) == 0:
            return None
        df = df.with_columns(
            _apply_transform(df["value"], acfg.transform).alias("value")
        )

    # ── Enantiomer handling (applied after transform) ────────────────────────
    if acfg.enantiomer_handling in ("mean", "keep_smallest", "keep_largest"):
        inchi_ns = _canonical_inchi_no_stereo(df["canonical_smiles"])
        df = df.with_columns(inchi_ns.alias("_inchi_ns"))

        if acfg.enantiomer_handling == "mean":
            df = (
                df.group_by(["_inchi_ns", "assay_name"])
                .agg([
                    pl.col("value").mean(),
                    pl.col("canonical_smiles").first(),
                ])
            )
        elif acfg.enantiomer_handling == "keep_smallest":
            df = (
                df.sort("value")
                .group_by(["_inchi_ns", "assay_name"])
                .agg([
                    pl.col("value").first(),
                    pl.col("canonical_smiles").first(),
                ])
            )
        else:  # keep_largest
            df = (
                df.sort("value", descending=True)
                .group_by(["_inchi_ns", "assay_name"])
                .agg([
                    pl.col("value").first(),
                    pl.col("canonical_smiles").first(),
                ])
            )

        df = df.drop("_inchi_ns")

    # ── Outlier removal ──────────────────────────────────────────────────────
    if acfg.outlier_removal == "iqr":
        df = _remove_outliers_iqr(df)
    elif acfg.outlier_removal == "zscore":
        df = _remove_outliers_zscore(df)

    if len(df) == 0:
        return None

    return df.select(["canonical_smiles", "assay_name", "value"])
