#!/usr/bin/env python3
"""MMP pipeline CLI entry point.

Usage:
  python run.py build     --input data.csv --config config.yaml --output mmp.duckdb
  python run.py query-assay --db mmp.duckdb --assay solubility --top 50
  python run.py query-mol   --db mmp.duckdb --smiles "c1ccccc1OC" --assay solubility
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mmp.run")


# ── build ────────────────────────────────────────────────────────────────────

def cmd_build(args: argparse.Namespace) -> int:
    import polars as pl
    from tqdm import tqdm
    from mmp.config import load_config
    from mmp.preprocessing import preprocess
    from mmp.fragmentation import fragment_molecules
    from mmp.indexing import generate_pairs
    from mmp.statistics import (
        compute_pair_deltas,
        aggregate_transform_stats,
        aggregate_fragment_influence,
    )
    from mmp import database as db

    steps = tqdm(
        ["config", "read", "preprocess", "fragment", "pairs",
         "deltas", "transform_stats", "frag_influence", "write_db"],
        desc="Pipeline",
        unit="step",
    )

    steps.set_postfix_str("loading config")
    cfg = load_config(args.config)
    steps.update(1)

    steps.set_postfix_str("reading CSV")
    df = pl.read_csv(args.input, infer_schema_length=10000)
    logger.info("Input: %d rows, columns: %s", len(df), df.columns)
    steps.update(1)

    steps.set_postfix_str("preprocessing")
    compounds = preprocess(df, cfg)
    logger.info("After preprocessing: %d (mol_id, assay, value) rows", len(compounds))
    steps.update(1)

    steps.set_postfix_str("fragmenting")
    fragments = fragment_molecules(compounds, cfg.global_config)
    logger.info("Fragments generated: %d", len(fragments))
    steps.update(1)

    if len(fragments) == 0:
        steps.close()
        logger.error("No fragments generated — check your input data and config filters.")
        return 1

    steps.set_postfix_str("generating pairs")
    pairs = generate_pairs(fragments)
    logger.info("Raw pairs: %d", len(pairs))
    steps.update(1)

    steps.set_postfix_str("computing deltas")
    pair_values = compute_pair_deltas(pairs, compounds)
    logger.info("Pair-value rows: %d", len(pair_values))
    steps.update(1)

    steps.set_postfix_str("transform stats")
    transform_stats = aggregate_transform_stats(pair_values, cfg)
    logger.info("Transform stats: %d rows", len(transform_stats))
    steps.update(1)

    steps.set_postfix_str("fragment influence")
    frag_influence = aggregate_fragment_influence(pair_values, cfg)
    logger.info("Fragment influence: %d rows", len(frag_influence))
    steps.update(1)

    steps.set_postfix_str("writing database")
    con = db.create_database(args.output)
    db.write_compounds(con, compounds)
    db.write_assay_values(con, compounds)
    db.write_fragments(con, fragments)
    db.write_pairs(con, pair_values)
    db.write_transform_stats(con, transform_stats)
    db.write_fragment_influence(con, frag_influence)
    con.close()
    steps.update(1)

    steps.set_postfix_str("done")
    steps.close()
    logger.info("Done. Database written to %s", args.output)
    return 0


# ── query-assay ──────────────────────────────────────────────────────────────

def cmd_query_assay(args: argparse.Namespace) -> int:
    from mmp.query import top_fragments_for_assay, top_transforms_for_assay

    if args.mode == "fragments":
        result = top_fragments_for_assay(
            db_path=args.db,
            assay_name=args.assay,
            top_n=args.top,
            min_pairs=args.min_pairs,
            sort_by=args.sort_by,
        )
    else:
        result = top_transforms_for_assay(
            db_path=args.db,
            assay_name=args.assay,
            top_n=args.top,
            min_pairs=args.min_pairs,
        )

    if len(result) == 0:
        print(f"No results for assay '{args.assay}'")
        return 0

    with __import__("polars").Config(tbl_rows=args.top, tbl_width_chars=200):
        print(result)
    return 0


# ── query-mol ────────────────────────────────────────────────────────────────

def cmd_query_mol(args: argparse.Namespace) -> int:
    from mmp.query import query_molecule

    result = query_molecule(
        db_path=args.db,
        smiles=args.smiles,
        assay_name=args.assay,
        max_results=args.max_results,
    )

    if len(result) == 0:
        print(f"No predictions found for '{args.smiles}'")
        return 0

    with __import__("polars").Config(tbl_rows=args.max_results, tbl_width_chars=200):
        print(result)
    return 0


# ── Argument parsing ─────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmp",
        description="Matched Molecular Pairs analysis pipeline",
    )
    parser.add_argument(
        "--no-quiet-rdkit",
        action="store_true",
        default=False,
        help="Show RDKit warnings (they are silenced by default)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── build ────────────────────────────────────────────────────────────────
    p_build = sub.add_parser("build", help="Build MMP database from input CSV")
    p_build.add_argument("--input", "-i", required=True, help="Input CSV file")
    p_build.add_argument("--config", "-c", default=None, help="YAML config file")
    p_build.add_argument("--output", "-o", required=True, help="Output DuckDB file path")

    # ── query-assay ──────────────────────────────────────────────────────────
    p_qa = sub.add_parser("query-assay", help="Query top fragments or transforms for an assay")
    p_qa.add_argument("--db", required=True, help="DuckDB database path")
    p_qa.add_argument("--assay", required=True, help="Assay name")
    p_qa.add_argument("--top", type=int, default=50, help="Number of results (default: 50)")
    p_qa.add_argument("--min-pairs", type=int, default=3, help="Minimum pairs filter")
    p_qa.add_argument(
        "--mode",
        choices=["fragments", "transforms"],
        default="transforms",
        help="Return top fragments or transforms (default: transforms)",
    )
    p_qa.add_argument(
        "--sort-by",
        default="mean_delta_introduced",
        help="Sort key for fragment mode",
    )

    # ── query-mol ────────────────────────────────────────────────────────────
    p_qm = sub.add_parser("query-mol", help="Predict property changes for a molecule")
    p_qm.add_argument("--db", required=True, help="DuckDB database path")
    p_qm.add_argument("--smiles", required=True, help="Query molecule SMILES")
    p_qm.add_argument("--assay", default=None, help="Restrict to one assay (optional)")
    p_qm.add_argument("--max-results", type=int, default=100, help="Max results (default: 100)")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    from mmp import silence_rdkit
    silence_rdkit(not args.no_quiet_rdkit)

    if args.command == "build":
        return cmd_build(args)
    elif args.command == "query-assay":
        return cmd_query_assay(args)
    elif args.command == "query-mol":
        return cmd_query_mol(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
