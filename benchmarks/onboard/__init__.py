"""DeepCTR model-onboarding pipeline.

A standalone CLI that standardizes the four stages of adding a new CTR model to
DeepCTR so the project can keep absorbing recent industry/academic models:

    discover  -> find & rank candidate models (knowledge base + optional web sweep)
    scaffold  -> generate model/test skeletons AND auto-wire every registration point
    audit     -> verify a model is fully wired (catches OneTrans-style half-integration)
    verify    -> prove correctness (unit test + audit) and effectiveness (benchmark AUC)
    docs      -> update README / Features / rst / History / RESULTS

Run with:  python -m benchmarks.onboard <command> [options]
"""
from __future__ import absolute_import, division, print_function

import os

# DeepCTR targets the Keras 2 API; force legacy Keras on TF>=2.16 before any TF import.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

# Repository root (…/DeepCTR), derived from this file's location.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

__all__ = ["REPO_ROOT"]
