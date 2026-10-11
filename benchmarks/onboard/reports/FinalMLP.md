# Verify report: FinalMLP

- category: **single**
- correctness:
    - audit (fully wired): PASS
    - unit test: PASS (2 passed, 3 warnings in 12.81s)
- effectiveness (primary metric = AUC):
    - DeepFM: AUC=0.5771 LogLoss=0.7297 [ok] (baseline)
    - FinalMLP: AUC=0.4158 LogLoss=0.6827 [ok]  <- new model
- paper-reported: {'dataset': 'Criteo_x1', 'AUC': 0.8137}

## Verdict

correctness PASS. FinalMLP AUC=0.4158; DeepFM baseline AUC=0.5771 (new model is below); paper AUC=0.8137.
