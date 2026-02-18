"""Matched Molecular Pairs (MMP) analysis pipeline."""

__version__ = "0.1.0"


def silence_rdkit(silent: bool = True) -> None:
    """Suppress RDKit C++ warning/info messages written to stderr."""
    if not silent:
        return
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
