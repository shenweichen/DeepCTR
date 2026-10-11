"""Dataset loaders for the benchmark suite.

Each loader returns one of the typed containers from ``benchmarks.common``
(``SingleTaskData`` / ``MultiTaskData`` / ``SequenceData``) with feature
columns and ready-to-fit train/test inputs already built.
"""
from .census import load_census
from .criteo import load_criteo
from .sequence import make_sequence_datasets

__all__ = ["load_criteo", "load_census", "make_sequence_datasets"]
