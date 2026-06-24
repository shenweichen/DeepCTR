# Model onboarding pipeline (`benchmarks.onboard`)

A standalone CLI that standardizes adding a new CTR model to DeepCTR across four
stages, so the project can keep absorbing recent industry/academic models:

```
discover  ->  scaffold  ->  (implement core)  ->  verify  ->  docs
```

Run everything CPU-only with legacy Keras (DeepCTR targets the Keras 2 API):

```bash
export CUDA_VISIBLE_DEVICES=""
export TF_USE_LEGACY_KERAS=1        # requires `pip install tf-keras==<your TF x.y>` on TF>=2.16
```

## Commands

| Command | What it does |
| --- | --- |
| `discover --list` | List/rank candidate models from `candidates.json`; marks which are already implemented. |
| `discover --refresh` | Print a research prompt to refresh the candidate KB using your own web-search capability. |
| `scaffold <Name> --category single\|sequence\|multitask` | Generate model + test skeletons and **auto-wire all 6 registration points** (`deepctr/models/__init__.py` import + `__all__`, sub-package `__init__`, `deepctr/layers/__init__.py` `custom_objects`, `benchmarks/registry.py`). Use `--with-layer` to also create a custom layer, `--wire-only` for an already-written model. |
| `audit [--name X ...]` | Check every discovered model is fully wired (importable, in `__all__`, custom layers registered, has a test, has a registry builder). Catches half-integrated models. |
| `verify <Name>` | Correctness (unit test + audit) **and** effectiveness (benchmark AUC vs an in-track baseline, compared to the paper number). Writes `reports/<Name>.md`. |
| `docs <Name>` | Idempotently update `README.md`, `docs/source/Features.md`, the autodoc `.rst` + toctree, `History.md`, and `RESULTS.md`. |
| `onboard <Name> --category ...` | `scaffold` → (pause to implement the core) → `verify` → `docs`. |

## Typical flow for a new model

```bash
python -m benchmarks.onboard discover --list
python -m benchmarks.onboard scaffold FinalMLP --category single   # generates + wires
#   implement the model core in deepctr/models/finalmlp.py (replace the TODO block)
python -m benchmarks.onboard verify FinalMLP                        # correctness + effectiveness
python -m benchmarks.onboard docs   FinalMLP                        # update all docs
python -m benchmarks.onboard audit                                  # confirm 100% wired
```

## Notes

- `candidates.json` is the version-controlled knowledge base of candidate models.
- `reports/<Name>.md` are kept as artifacts; transient benchmark leaderboards go
  to the gitignored `benchmarks/results/`.
- Scaffolded test files use a generic signature; adjust the generated
  `tests/models/<Name>_test.py` if your model's constructor differs (e.g. FinalMLP).
