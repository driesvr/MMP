"""Pair generation via Polars self-join on constant_smi."""

from __future__ import annotations

import logging

import polars as pl

logger = logging.getLogger(__name__)

MAX_BUCKET_SIZE = 1000  # cap to avoid O(n^2) blowup in huge buckets

# Schema for the pairs DataFrame
PAIR_SCHEMA = {
    "mol_id_left": pl.UInt32,
    "mol_id_right": pl.UInt32,
    "constant_smi": pl.Utf8,
    "variable_left": pl.Utf8,
    "variable_right": pl.Utf8,
    "num_cuts": pl.UInt8,
    "attach_env_left": pl.Utf8,
    "attach_env_right": pl.Utf8,
    "transform_smirks": pl.Utf8,
}


def generate_pairs(
    fragments: pl.DataFrame,
    max_bucket_size: int = MAX_BUCKET_SIZE,
) -> pl.DataFrame:
    """Self-join fragment table on constant_smi to produce all MMP pairs.

    Parameters
    ----------
    fragments:
        Output of ``fragmentation.fragment_molecules``.
    max_bucket_size:
        Buckets (constant_smi groups) larger than this are sampled down to
        avoid quadratic blowup.

    Returns
    -------
    Polars DataFrame with columns matching PAIR_SCHEMA.
    """
    if len(fragments) == 0:
        return pl.DataFrame(schema=PAIR_SCHEMA)

    # ── Cap oversized buckets ────────────────────────────────────────────────
    bucket_sizes = (
        fragments.group_by("constant_smi")
        .agg(pl.len().alias("n"))
    )
    large_buckets = bucket_sizes.filter(pl.col("n") > max_bucket_size)["constant_smi"].to_list()

    if large_buckets:
        logger.warning(
            "%d constant_smi buckets exceed max_bucket_size=%d and will be sampled",
            len(large_buckets),
            max_bucket_size,
        )
        # Sample each large bucket down to max_bucket_size
        small_part = fragments.filter(~pl.col("constant_smi").is_in(large_buckets))
        large_parts = []
        for csmi in large_buckets:
            bucket = fragments.filter(pl.col("constant_smi") == csmi)
            large_parts.append(bucket.sample(n=max_bucket_size, seed=42))
        fragments = pl.concat([small_part] + large_parts)

    # ── Self-join on constant_smi ────────────────────────────────────────────
    left = fragments.select([
        pl.col("mol_id").alias("mol_id_left"),
        "constant_smi",
        pl.col("variable_smi").alias("variable_left"),
        pl.col("num_cuts"),
        pl.col("attach_env").alias("attach_env_left"),
    ])
    right = fragments.select([
        pl.col("mol_id").alias("mol_id_right"),
        "constant_smi",
        pl.col("variable_smi").alias("variable_right"),
        pl.col("num_cuts").alias("num_cuts_right"),
        pl.col("attach_env").alias("attach_env_right"),
    ])

    pairs = (
        left.join(right, on="constant_smi", how="inner")
        # Remove self-pairs and keep only (i < j) to avoid duplicates
        .filter(pl.col("mol_id_left") < pl.col("mol_id_right"))
        # num_cuts must match between both fragments of the pair
        .filter(pl.col("num_cuts") == pl.col("num_cuts_right"))
        .drop("num_cuts_right")
    )

    if len(pairs) == 0:
        return pl.DataFrame(schema=PAIR_SCHEMA)

    # ── Canonicalize transform SMIRKS: sort LHS and RHS lexicographically ───
    pairs = _canonicalize_transforms(pairs)

    return pairs.select(list(PAIR_SCHEMA.keys()))


def _canonicalize_transforms(pairs: pl.DataFrame) -> pl.DataFrame:
    """Ensure transform_smirks is always written as min(A,B) >> max(A,B).

    When we swap variable_left and variable_right, we also swap mol_id_left
    and mol_id_right so that delta = value_right - value_left is consistent.
    """
    # Compute SMIRKS for both orientations
    pairs = pairs.with_columns(
        pl.concat_str([pl.col("variable_left"), pl.lit(">>"), pl.col("variable_right")])
        .alias("_smirks_fwd"),
        pl.concat_str([pl.col("variable_right"), pl.lit(">>"), pl.col("variable_left")])
        .alias("_smirks_rev"),
    )

    # Determine which orientation is canonical (alphabetically smaller)
    need_swap = pl.col("_smirks_fwd") > pl.col("_smirks_rev")

    pairs = pairs.with_columns([
        pl.when(need_swap).then(pl.col("mol_id_right")).otherwise(pl.col("mol_id_left"))
        .alias("mol_id_left"),
        pl.when(need_swap).then(pl.col("mol_id_left")).otherwise(pl.col("mol_id_right"))
        .alias("mol_id_right"),
        pl.when(need_swap).then(pl.col("variable_right")).otherwise(pl.col("variable_left"))
        .alias("variable_left"),
        pl.when(need_swap).then(pl.col("variable_left")).otherwise(pl.col("variable_right"))
        .alias("variable_right"),
        pl.when(need_swap).then(pl.col("attach_env_right")).otherwise(pl.col("attach_env_left"))
        .alias("attach_env_left"),
        pl.when(need_swap).then(pl.col("attach_env_left")).otherwise(pl.col("attach_env_right"))
        .alias("attach_env_right"),
        pl.when(need_swap).then(pl.col("_smirks_rev")).otherwise(pl.col("_smirks_fwd"))
        .alias("transform_smirks"),
    ])

    return pairs.drop(["_smirks_fwd", "_smirks_rev"])
