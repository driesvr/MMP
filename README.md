# MMP Explorer

A matched molecular pairs (MMP) analysis pipeline with an interactive web UI for exploring property-change predictions.

## What it does

Fragments molecules using rule-based bond SMARTS (exocyclic ring bonds and heteroatom–sp3-carbon bonds), pairs them by shared constant scaffold, computes property-change statistics across all observed transformations, and stores results in DuckDB. The web UI lets you draw or paste a SMILES, click any substitutable bond, and see ranked proposals for what to swap in — with predicted Δproperty, confidence (n pairs), and Q1–Q3 spread.

## Quick start

```bash
# Install dependencies
pip install rdkit polars duckdb pyyaml fastapi uvicorn pydantic

# Build a database from CSV (columns: smiles, assay_name, value)
python run.py build -i data.csv -o mmp.duckdb

# Start the web UI
python serve.py --db-dir . --port 8000
# → http://127.0.0.1:8000
```

## CLI

```bash
# Top fragments for an assay
python run.py query-assay --db mmp.duckdb --assay logS --top 50

# Per-molecule predictions
python run.py query-mol --db mmp.duckdb --smiles "CC(=O)Nc1ccc(Cl)cc1" --assay logS
```

## Input CSV format

| column | description |
|--------|-------------|
| `smiles` | molecule SMILES (any notation, canonicalized internally) |
| `assay_name` | string identifier for the property |
| `value` | numeric measurement |

Multiple rows per molecule (different assays) are fine. Qualifiers (`>`, `<`) are stripped by default.

## Configuration

Pass with `--config config.yaml`. All fields are optional; the defaults shown below are used for anything omitted. Named assays inherit from `_default`, which itself inherits from the built-in defaults.

```yaml
global:
  max_heavy_atoms: 70          # molecules larger than this are skipped entirely
  max_cuts: 1                  # 1 = single-cut only; 2 = also generate double-cut pairs
  max_variable_heavy_atoms: 13 # R-group (variable fragment) size limit in heavy atoms
  min_constant_heavy_atoms: 5  # scaffold (constant fragment) must be at least this large
  variable_to_constant_ratio: 0.33  # variable must be smaller than this fraction of the molecule
  n_workers: -1                # worker processes for fragmentation (-1 = all CPUs, 1 = serial)

assays:
  my_assay:
    transform: none            # value pre-processing applied before pairing:
                               #   none       — use values as-is
                               #   log10      — log10(value); non-positive values are dropped
                               #   neg_log10  — -log10(value); non-positive values are dropped
                               #   pIC50      — 6 - log10(value); assumes value in µM

    qualifier_handling: strip  # how to handle qualifier prefixes (>, <, >=, <=):
                               #   strip — remove the qualifier prefix and keep the numeric part.
                               #           Works for both embedded qualifiers in the value string
                               #           (e.g. ">5.0" becomes 5.0) and a separate 'qualifier'
                               #           column. This is the recommended default.
                               #   drop  — discard any row that carries a qualifier (either
                               #           embedded in the value string or in a separate column)
                               #
                               # Input CSV may optionally include a separate 'qualifier' column
                               # alongside a numeric 'value' column; both formats are supported.

    enantiomer_handling: mean  # how to collapse stereoisomers of the same flat structure.
                               # Grouping is by InChI with stereo layers stripped (/SNon).
                               # All deduplication happens after the transform is applied.
                               #   keep          — treat each stereoisomer as a distinct compound
                               #   mean          — average the (transformed) values
                               #   keep_smallest — keep the enantiomer with the lowest value
                               #   keep_largest  — keep the enantiomer with the highest value

    outlier_removal: none      # per-assay outlier filtering, applied after transform:
                               #   none   — no filtering
                               #   iqr    — remove values outside Q1 - 1.5×IQR … Q3 + 1.5×IQR
                               #   zscore — remove values with |z-score| > 3

    min_pairs: 3               # minimum number of matched pairs required to report a transform

  _default:                    # fallback for any assay not explicitly listed
    transform: none
    qualifier_handling: strip
    enantiomer_handling: mean
    outlier_removal: none
    min_pairs: 3
```

## Architecture

```
CSV → preprocessing → fragmentation → indexing → statistics → DuckDB
```

- **fragmentation** (`mmp/fragmentation.py`) — two bond SMARTS patterns (exocyclic ring bonds; heteroatom–sp3-carbon bonds), single-cut, parallel via `ProcessPoolExecutor`
- **indexing** (`mmp/indexing.py`) — self-join on constant scaffold, caps bucket size at 1000
- **statistics** (`mmp/statistics.py`) — per-transform aggregation: mean, std, median, Q1, Q3, min, max
- **query** (`mmp/query.py`) — looks up transforms by variable fragment across all scaffolds; filters by attachment atom type (C_SP2, C_SP3, O_SP2, …) to avoid cross-context mixing
- **web** (`mmp/web.py`) — FastAPI, fragment the query mol with vector tracking, SVG rendering, proposals grouped by bond

## Running tests

```bash
pytest tests/
```

## Database schema (DuckDB)

Six tables: `compounds`, `assay_values`, `fragments`, `pairs`, `transform_stats`, `fragment_influence`. See `mmp/database.py` for full DDL.
