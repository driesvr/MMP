"""YAML config loader and validation for MMP pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml

# ── Valid option sets ────────────────────────────────────────────────────────

VALID_TRANSFORMS = {"none", "log10", "pIC50", "neg_log10"}
VALID_QUALIFIER_HANDLING = {"strip", "drop"}
VALID_ENANTIOMER_HANDLING = {"keep", "mean", "keep_smallest", "keep_largest"}
VALID_OUTLIER_REMOVAL = {"none", "iqr", "zscore"}


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class GlobalConfig:
    max_heavy_atoms: int = 70
    max_cuts: int = 2
    max_variable_heavy_atoms: int = 13
    min_constant_heavy_atoms: int = 5
    variable_to_constant_ratio: float = 0.33
    n_workers: int = -1


@dataclass
class AssayConfig:
    transform: str = "none"
    qualifier_handling: str = "strip"
    enantiomer_handling: str = "mean"
    outlier_removal: str = "none"
    min_pairs: int = 3


@dataclass
class Config:
    global_config: GlobalConfig = field(default_factory=GlobalConfig)
    assays: dict[str, AssayConfig] = field(default_factory=dict)
    _default: AssayConfig = field(default_factory=AssayConfig)


# ── Validation helpers ───────────────────────────────────────────────────────

def _validate_assay(name: str, raw: dict[str, Any]) -> AssayConfig:
    allowed_keys = {
        "transform", "qualifier_handling", "enantiomer_handling",
        "outlier_removal", "min_pairs",
    }
    unknown = set(raw) - allowed_keys
    if unknown:
        raise ValueError(f"Assay '{name}' has unknown keys: {unknown}")

    transform = raw.get("transform", "none")
    if transform not in VALID_TRANSFORMS:
        raise ValueError(
            f"Assay '{name}': transform '{transform}' not in {VALID_TRANSFORMS}"
        )

    qh = raw.get("qualifier_handling", "strip")
    if qh not in VALID_QUALIFIER_HANDLING:
        raise ValueError(
            f"Assay '{name}': qualifier_handling '{qh}' not in {VALID_QUALIFIER_HANDLING}"
        )

    eh = raw.get("enantiomer_handling", "mean")
    if eh not in VALID_ENANTIOMER_HANDLING:
        raise ValueError(
            f"Assay '{name}': enantiomer_handling '{eh}' not in {VALID_ENANTIOMER_HANDLING}"
        )

    orv = raw.get("outlier_removal", "none")
    if orv not in VALID_OUTLIER_REMOVAL:
        raise ValueError(
            f"Assay '{name}': outlier_removal '{orv}' not in {VALID_OUTLIER_REMOVAL}"
        )

    min_pairs = raw.get("min_pairs", 3)
    if not isinstance(min_pairs, int) or min_pairs < 1:
        raise ValueError(f"Assay '{name}': min_pairs must be a positive integer")

    return AssayConfig(
        transform=transform,
        qualifier_handling=qh,
        enantiomer_handling=eh,
        outlier_removal=orv,
        min_pairs=min_pairs,
    )


def _validate_global(raw: dict[str, Any]) -> GlobalConfig:
    g = GlobalConfig()
    if "max_heavy_atoms" in raw:
        g.max_heavy_atoms = int(raw["max_heavy_atoms"])
    if "max_cuts" in raw:
        v = int(raw["max_cuts"])
        if v not in (1, 2):
            raise ValueError("global.max_cuts must be 1 or 2")
        g.max_cuts = v
    if "max_variable_heavy_atoms" in raw:
        g.max_variable_heavy_atoms = int(raw["max_variable_heavy_atoms"])
    if "min_constant_heavy_atoms" in raw:
        g.min_constant_heavy_atoms = int(raw["min_constant_heavy_atoms"])
    if "variable_to_constant_ratio" in raw:
        g.variable_to_constant_ratio = float(raw["variable_to_constant_ratio"])
    if "n_workers" in raw:
        g.n_workers = int(raw["n_workers"])
    return g


# ── Public API ───────────────────────────────────────────────────────────────

def load_config(path: str | None = None) -> Config:
    """Load and validate a YAML config file.

    If *path* is None, a default Config is returned.
    """
    if path is None:
        return Config()

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    cfg = Config()

    if "global" in raw:
        cfg.global_config = _validate_global(raw["global"])

    default_raw: dict[str, Any] = {}
    assays_raw: dict[str, Any] = raw.get("assays", {})

    if "_default" in assays_raw:
        default_raw = assays_raw.pop("_default")
        cfg._default = _validate_assay("_default", default_raw)

    for name, assay_raw in assays_raw.items():
        # Merge _default under explicit assay values
        merged = {**default_raw, **assay_raw}
        cfg.assays[name] = _validate_assay(name, merged)

    return cfg


def get_assay_config(cfg: Config, assay_name: str) -> AssayConfig:
    """Return the AssayConfig for *assay_name*, falling back to _default."""
    return cfg.assays.get(assay_name, cfg._default)
