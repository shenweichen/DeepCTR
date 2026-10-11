# Verify report: MaskNet

- category: **single**
- correctness:
    - audit (fully wired): PASS
    - unit test: PASS (2 passed, 3 warnings in 18.14s)
- effectiveness (primary metric = AUC):
    - DeepFM: AUC=0.5771 LogLoss=0.7297 [ok] (baseline)
    - MaskNet: AUC=0.552 LogLoss=0.526 [ok]  <- new model
- paper-reported: {'dataset': 'Criteo_x1', 'AUC': 0.8124}

## Verdict

correctness PASS. MaskNet AUC=0.552; DeepFM baseline AUC=0.5771 (new model is below); paper AUC=0.8124.
