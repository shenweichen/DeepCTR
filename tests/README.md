# DeepCTR correctness testing

Passing a training smoke test is necessary, but it does not prove that a paper
equation was implemented correctly.  DeepCTR uses the following reusable
correctness ladder.

| Level | Contract | What it catches |
| --- | --- | --- |
| C0 | import/build/shape | missing wiring and incompatible signatures |
| C1 | tiny train + finite inference | broken forward or optimizer paths |
| C2 | weight and full-model round-trip with equal predictions | incomplete config/custom-object serialization |
| C3 | independent NumPy reference on deterministic tensors | wrong axes, reductions, head splits, residuals, and equations |
| C4 | domain invariants and boundary cases | masking, causality, symmetry, padding, empty/minimal inputs |
| C5 | finite and numerical gradients | disconnected or unstable differentiation paths |
| C6 | official-code differential / paper-scale reproduction | architectural or preprocessing drift |

## Minimum contracts

- Every model test calls `tests.utils.check_model` (or
  `tests.utils_mtl.check_mtl_model`) for C0-C2. These helpers compare predictions
  after both weight-only and complete-model serialization; merely loading
  without an exception is not sufficient.
- Correctness fixtures are deterministic. Call `set_test_seed` before generating
  random inputs so a numerical failure can be reproduced exactly.
- Every new mathematical custom layer has at least one C3 reference test using
  `tests.correctness.assert_forward_matches` and one C5 gradient test.
- Sequence, masking, set, and interaction layers add the relevant C4 invariant.
- C6 is required before claiming paper reproduction. Benchmark AUC alone is an
  effectiveness signal, not proof of equation-level correctness.

## Reference-test pattern

```python
import numpy as np

from tests.correctness import assert_finite_gradients, assert_forward_matches


def reference(inputs, weights):
    # Translate the paper equation independently. Do not call the layer or copy
    # its TensorFlow expression into NumPy line-for-line.
    return np.matmul(inputs, weights["kernel"]) + weights["bias"]


def test_my_layer_reference_and_gradients():
    x = np.asarray([[0.2, -0.1]], dtype="float32")
    layer = MyLayer()
    deterministic = {
        "kernel": np.asarray([[0.3], [0.7]], dtype="float32"),
        "bias": np.asarray([0.1], dtype="float32"),
    }
    assert_forward_matches(layer, x, reference, weights=deterministic)
    assert_finite_gradients(layer, x)
```

For tiny differentiable tensors, also use `assert_numerical_gradient`.  For
semantic properties use `assert_invariant`, selecting only the output region
that should remain unchanged (for example, a causal token before a modified
future token).

Concrete examples live in `tests/layers/*_correctness_test.py` and cover a
bilinear equation, a composite interaction block, an exact attention mask, and
causal invariance.

## Running the contracts

```bash
CUDA_VISIBLE_DEVICES="" TF_USE_LEGACY_KERAS=1 \
  python -m pytest tests/layers/*_correctness_test.py -q

CUDA_VISIBLE_DEVICES="" TF_USE_LEGACY_KERAS=1 \
  python -m pytest tests/ -q
```
