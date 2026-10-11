# Verify report: WuKong

- category: **single**
- correctness:
    - audit (fully wired): PASS
    - unit test: PASS (2 passed, 3 warnings in 18.12s)
- effectiveness (primary metric = AUC):
    - DeepFM: AUC=0.5771 LogLoss=0.7297 [ok] (baseline)
    - WuKong: AUC=0.5627 LogLoss=0.5764 [ok]  <- new model

## Verdict

correctness PASS. WuKong AUC=0.5627; DeepFM baseline AUC=0.5771 (new model is below).
