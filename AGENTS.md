# AGENTS.md

Guidance for AI agents (Codex, Claude Code, etc.) working in this repo.

## Environment setup (required before running tests or benchmarks)

DeepCTR targets the **Keras 2** API. This box ships TF with Keras 3, so you must
install legacy Keras and force it on, or every model test fails on label-rank /
serialization errors.

```bash
pip install pytest                       # not preinstalled
pip install "tf-keras==<TF major.minor>" # e.g. tf-keras==2.19.0 for TF 2.19; required on TF>=2.16
```

Prefix every test / benchmark / onboarding command with:

```bash
CUDA_VISIBLE_DEVICES="" TF_USE_LEGACY_KERAS=1 <command>
```

(`CUDA_VISIBLE_DEVICES=""` avoids a GPU CUDA/PTX mismatch; `TF_USE_LEGACY_KERAS=1`
selects the Keras-2 backend.)

## Running tests

```bash
CUDA_VISIBLE_DEVICES="" TF_USE_LEGACY_KERAS=1 python -m pytest tests/ -q
CUDA_VISIBLE_DEVICES="" TF_USE_LEGACY_KERAS=1 python -m pytest tests/models/DeepFM_test.py -q   # single model
```

## Adding a new model — use the onboarding pipeline

The `benchmarks.onboard` CLI standardizes adding a new CTR model across four
stages. Full reference: `benchmarks/onboard/README.md`.

```bash
export CUDA_VISIBLE_DEVICES="" TF_USE_LEGACY_KERAS=1

python -m benchmarks.onboard discover --list                       # candidate models
python -m benchmarks.onboard scaffold <Name> --category single     # codegen + auto-wire all registration points
#   -> implement the model core in deepctr/models/<name>.py (replace the TODO block)
python -m benchmarks.onboard verify <Name>                         # unit test + audit + benchmark vs baseline
python -m benchmarks.onboard docs   <Name>                         # update README/Features/rst/History/RESULTS
python -m benchmarks.onboard audit                                 # confirm every model is fully wired
```

Categories: `single` | `sequence` | `multitask`. Add `--with-layer` to also
scaffold a custom layer, `--wire-only` for an already-written model.

### Gotchas

- `scaffold` generates `tests/models/<Name>_test.py` with a **generic** constructor
  signature (`dnn_hidden_units=...`). If your model's signature differs, edit the
  generated test to pass the right args (e.g. FinalMLP uses `mlp1_hidden_units`).
- Every custom layer must end up in `deepctr/layers/__init__.py` `custom_objects`
  (scaffold does this), create sub-layers in `__init__` not `build`, avoid
  `Lambda(lambda ...)` (use a small serializable layer), and not collide on class
  name with another registered layer — or `save_model`/`load_model` will fail.
  Run `python -m benchmarks.onboard audit` to verify wiring.

## Conventions

- Models are factory functions returning a `tf.keras.Model`; see `deepctr/models/wdl.py`
  (single), `deepctr/models/sequence/din.py` (sequence), `deepctr/models/multitask/mmoe.py`.
- Use `tensorflow.keras` imports (not `tensorflow.python.keras`).
- Commit only when asked; branch off `master` for PRs.
