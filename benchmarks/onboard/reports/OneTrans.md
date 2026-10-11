# Verify report: OneTrans

- category: **sequence**
- correctness:
    - audit (fully wired): PASS
    - unit test: PASS (1 passed, 2 warnings in 10.16s)
- effectiveness (primary metric = AUC):
    - OneTrans: AUC=0.4735 LogLoss=0.708 [ok]  <- new model

## Verdict

correctness PASS. OneTrans AUC=0.4735.
