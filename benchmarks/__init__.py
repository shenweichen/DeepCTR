"""DeepCTR model evaluation / benchmark suite.

A small, self-contained harness that trains and compares DeepCTR models
end-to-end across three tracks (single-task CTR, multitask, sequence) and
emits a sorted leaderboard.

Run as a module::

    python -m benchmarks.benchmark --track single

See ``benchmarks/README.md`` for full usage.
"""

import os as _os

# DeepCTR targets the Keras 2 API. On TensorFlow >= 2.16 (which defaults to
# Keras 3) the sequence/multitask layers break unless legacy Keras is used --
# exactly as the library's own CI does (TF_USE_LEGACY_KERAS=1 + the `tf-keras`
# package). Set it here, before any submodule imports TensorFlow, so every
# entry point (CLI, tests, direct imports) is consistent. Users can override by
# exporting the variable themselves.
_os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

__all__ = []
