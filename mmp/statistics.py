"""Delta computation and aggregation for MMP pairs."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from mmp.config import Config

logger = logging.getLogger(__name__)


def compute_pair_deltas(
    pairs: pl.DataFrame,
    compounds: pl.DataFrame,
) -> pl.DataFrame:
    """Join pairs with assay values and compute property deltas.

    Parameters
    ----------
    pairs:
        Output of ``indexing.generate_pairs``.
    compounds:
        DataFrame with columns ``[mol_id, canonical_smiles, assay_name, value]``.

    Returns
    -------
    DataFrame with pair columns plus ``assay_name`` and ``delta``.
    """
    if len(pairs) == 0 or len(compounds) == 0:
        return pl.DataFrame()

    # Join left molecule values
    pair_values = pairs.join(
        compounds.select(["mol_id", "assay_name", "value"]),
        left_on="mol_id_left",
        right_on="mol_id",
        how="inner",
    )

    # Join right molecule values (must share the same assay)
    pair_values = pair_values.join(
        compounds.select(["mol_id", "assay_name", "value"]),
        left_on=["mol_id_right", "assay_name"],
        right_on=["mol_id", "assay_name"],
        how="inner",
        suffix="_right",
    )

    # delta = value_right - value_left (in canonical direction: A>>B)
    pair_values = pair_values.with_columns(
        (pl.col("value_right") - pl.col("value")).alias("delta")
    ).rename({"value": "value_left"})

    return pair_values


def aggregate_transform_stats(
    pair_values: pl.DataFrame,
    cfg: "Config",
) -> pl.DataFrame:
    """Aggregate per (transform_smirks, assay_name).

    Returns
    -------
    DataFrame with columns:
        transform_smirks, assay_name, pair_count, mean_delta, std_delta,
        median_delta, min_delta, max_delta
    """
    if len(pair_values) == 0:
        return pl.DataFrame(schema={
            "transform_smirks": pl.Utf8,
            "assay_name": pl.Utf8,
            "pair_count": pl.UInt32,
            "mean_delta": pl.Float64,
            "std_delta": pl.Float64,
            "median_delta": pl.Float64,
            "min_delta": pl.Float64,
            "max_delta": pl.Float64,
            "q1_delta": pl.Float64,
            "q3_delta": pl.Float64,
        })

    stats = (
        pair_values
        .group_by(["transform_smirks", "assay_name"])
        .agg([
            pl.len().cast(pl.UInt32).alias("pair_count"),
            pl.col("delta").mean().alias("mean_delta"),
            pl.col("delta").std().alias("std_delta"),
            pl.col("delta").median().alias("median_delta"),
            pl.col("delta").min().alias("min_delta"),
            pl.col("delta").max().alias("max_delta"),
            pl.col("delta").quantile(0.25).alias("q1_delta"),
            pl.col("delta").quantile(0.75).alias("q3_delta"),
        ])
    )

    # Filter by min_pairs per assay
    from mmp.config import get_assay_config

    min_pairs_map: dict[str, int] = {}
    for assay_name in stats["assay_name"].unique().to_list():
        min_pairs_map[assay_name] = get_assay_config(cfg, assay_name).min_pairs

    min_pairs_series = pl.Series(
        "_min_pairs",
        [min_pairs_map[a] for a in stats["assay_name"].to_list()],
        dtype=pl.Int64,
    )
    stats = stats.with_columns(min_pairs_series)
    stats = stats.filter(pl.col("pair_count") >= pl.col("_min_pairs")).drop("_min_pairs")

    return stats


def aggregate_fragment_influence(
    pair_values: pl.DataFrame,
    cfg: "Config",
) -> pl.DataFrame:
    """Aggregate per (variable_smi, assay_name) — "fragment influence".

    Only considers rows where the fragment appears on the RIGHT side (introduced).

    Returns
    -------
    DataFrame with columns:
        variable_smi, assay_name, times_introduced,
        mean_delta_introduced, std_delta_introduced
    """
    if len(pair_values) == 0:
        return pl.DataFrame(schema={
            "variable_smi": pl.Utf8,
            "assay_name": pl.Utf8,
            "times_introduced": pl.UInt32,
            "mean_delta_introduced": pl.Float64,
            "std_delta_introduced": pl.Float64,
        })

    influence = (
        pair_values
        .rename({"variable_right": "variable_smi"})
        .group_by(["variable_smi", "assay_name"])
        .agg([
            pl.len().cast(pl.UInt32).alias("times_introduced"),
            pl.col("delta").mean().alias("mean_delta_introduced"),
            pl.col("delta").std().alias("std_delta_introduced"),
        ])
    )

    return influence
