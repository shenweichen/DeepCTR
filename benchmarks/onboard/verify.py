"""Validate a model two ways:

  correctness   - the model's unit test passes (compile / train / save+load
                  round-trip) AND `audit` reports it fully wired.
  effectiveness - a benchmark run trains it on data and its AUC is compared
                  against an in-track baseline (and the paper number if known).

Writes a Markdown report to benchmarks/onboard/reports/<name>.md.
"""
from __future__ import absolute_import, division, print_function

import os
import subprocess
import sys

from . import REPO_ROOT
from ._util import discover_models, test_file_for

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
# Transient benchmark leaderboards go to the gitignored results dir, keeping
# only the curated verify reports under REPORTS_DIR.
BENCH_OUT_DIR = os.path.join(REPO_ROOT, "benchmarks", "results")

TRACK_PRIMARY = {"single": "AUC", "sequence": "AUC", "multitask": "mean_AUC"}
DEFAULT_BASELINE = {"single": "DeepFM", "sequence": "DIN", "multitask": "SharedBottom"}


def _resolve_category(name, explicit):
    if explicit:
        return explicit
    info = discover_models().get(name)
    if not info:
        raise SystemExit("unknown model %r; pass --category" % name)
    return info["category"]


# --------------------------------------------------------------------------- #
# Correctness
# --------------------------------------------------------------------------- #
def _run_correctness(name):
    result = {"audit": None, "unit_test": None}

    from .audit import audit_models, _row_ok
    rows, _ = audit_models(only={name})
    result["audit"] = bool(rows) and _row_ok(rows[0])
    result["audit_row"] = rows[0] if rows else None

    test_path = test_file_for(name)
    if not test_path:
        result["unit_test"] = False
        result["unit_test_note"] = "no test file found"
        return result

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TF_USE_LEGACY_KERAS="1")
    print("  running unit test: %s" % os.path.relpath(test_path, REPO_ROOT))
    proc = subprocess.run([sys.executable, "-m", "pytest", test_path, "-q"],
                          cwd=REPO_ROOT, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = proc.stdout.decode("utf-8", "replace")
    result["unit_test"] = proc.returncode == 0
    # keep the pytest summary line(s)
    tail = [l for l in out.splitlines() if "passed" in l or "failed" in l or "error" in l.lower()]
    result["unit_test_note"] = tail[-1].strip() if tail else "see output"
    return result


# --------------------------------------------------------------------------- #
# Effectiveness
# --------------------------------------------------------------------------- #
def _run_benchmark(name, category, args):
    from benchmarks import benchmark

    track = category
    baseline = args.baseline if args.baseline and args.baseline != name else DEFAULT_BASELINE[category]
    if baseline == name:
        baseline = None

    models = name if baseline is None else "%s,%s" % (name, baseline)
    argv = ["--track", track, "--models", models,
            "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
            "--val-split", "0", "--output-dir", BENCH_OUT_DIR]
    if args.data_path:
        argv += ["--data-path", args.data_path]
        if category == "sequence":
            argv += ["--seq-source", "movielens"]
    elif category == "sequence":
        argv += ["--n-samples", "400" if args.quick else "1200"]

    print("  benchmark: python -m benchmarks.benchmark %s" % " ".join(argv))
    bargs = benchmark.build_parser().parse_args(argv)
    runner = {"single": benchmark.run_single,
              "sequence": benchmark.run_sequence,
              "multitask": benchmark.run_multitask}[track]
    results = runner(bargs)
    by_name = {r.model: r for r in results}
    return by_name, baseline


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _paper_metric(name):
    try:
        from .discover import load_candidates
        for c in load_candidates():
            if c.get("name") == name:
                return c.get("paper_metric")
    except Exception:
        pass
    return None


def _write_report(name, category, correctness, bench, baseline, verdict):
    if not os.path.isdir(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)
    primary = TRACK_PRIMARY[category]
    lines = ["# Verify report: %s" % name, "",
             "- category: **%s**" % category,
             "- correctness:",
             "    - audit (fully wired): %s" % ("PASS" if correctness["audit"] else "FAIL"),
             "    - unit test: %s (%s)" % ("PASS" if correctness["unit_test"] else "FAIL",
                                           correctness.get("unit_test_note", "")),
             "- effectiveness (primary metric = %s):" % primary]
    if bench is not None:
        for mname, r in bench.items():
            tag = "  <- new model" if mname == name else (" (baseline)" if mname == baseline else "")
            metrics = " ".join("%s=%s" % (k, v) for k, v in r.metrics.items()) if r.metrics else r.note
            lines.append("    - %s: %s [%s]%s" % (mname, metrics, r.status, tag))
    paper = _paper_metric(name)
    if paper:
        lines.append("- paper-reported: %s" % paper)
    lines += ["", "## Verdict", "", verdict, ""]
    path = os.path.join(REPORTS_DIR, name + ".md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def _make_verdict(name, category, correctness, bench, baseline):
    primary = TRACK_PRIMARY[category]
    ok = correctness["audit"] and correctness["unit_test"]
    parts = []
    parts.append("correctness %s" % ("PASS" if ok else "FAIL"))
    if bench is not None and name in bench and bench[name].status == "ok":
        new_auc = bench[name].metrics.get(primary)
        msg = "%s %s=%s" % (name, primary, new_auc)
        if baseline and baseline in bench and bench[baseline].status == "ok":
            base_auc = bench[baseline].metrics.get(primary)
            rel = "above" if (new_auc or 0) >= (base_auc or 0) else "below"
            msg += "; %s baseline %s=%s (new model is %s)" % (baseline, primary, base_auc, rel)
        paper = _paper_metric(name)
        if paper and isinstance(paper, dict) and paper.get("AUC"):
            msg += "; paper AUC=%s" % paper["AUC"]
        parts.append(msg)
    elif bench is not None:
        parts.append("benchmark did not produce a metric for %s" % name)
    return ". ".join(parts) + "."


def run(args):
    name = args.name
    category = _resolve_category(name, getattr(args, "category", None))
    print("Verifying %s (%s)" % (name, category))

    print("\n[1/2] correctness")
    correctness = _run_correctness(name)
    print("  audit: %s | unit test: %s"
          % ("PASS" if correctness["audit"] else "FAIL",
             "PASS" if correctness["unit_test"] else "FAIL"))

    bench, baseline = None, None
    if getattr(args, "skip_benchmark", False):
        print("\n[2/2] effectiveness: skipped (--skip-benchmark)")
    else:
        print("\n[2/2] effectiveness")
        try:
            bench, baseline = _run_benchmark(name, category, args)
        except Exception as e:
            print("  benchmark failed: %s" % e)

    verdict = _make_verdict(name, category, correctness, bench, baseline)
    path = _write_report(name, category, correctness, bench, baseline, verdict)
    print("\nVerdict: %s" % verdict)
    print("Report: %s" % os.path.relpath(path, REPO_ROOT))

    passed = correctness["audit"] and correctness["unit_test"]
    return 0 if passed else 1
