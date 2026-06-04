"""DeepCTR benchmark CLI.

Train and compare DeepCTR models end-to-end and emit a sorted leaderboard.

Examples
--------
    python -m benchmarks.benchmark --track single                 # bundled Criteo, all single-task models
    python -m benchmarks.benchmark --track single --download      # fetch Criteo DAC ~100k, then benchmark
    python -m benchmarks.benchmark --track single --data-path train.txt --hash --epochs 5
    python -m benchmarks.benchmark --track multitask
    python -m benchmarks.benchmark --track sequence --seq-source synthetic
    python -m benchmarks.benchmark --track all --quick
"""
from __future__ import absolute_import, division, print_function

import argparse
import os
import time

from .common import (ModelResult, _fmt, print_leaderboard, reset_session, set_seeds,
                     sort_results, write_results)
from .datasets import load_census, load_criteo, make_sequence_datasets
from .metrics import compute_metrics
from .registry import (MULTITASK_MODELS, SEQUENCE_MODELS, SINGLE_TASK_MODELS, SkipModel, select)

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(_PKG_DIR, "results")


def _short(exc):
    s = str(exc).strip().replace("\n", " ")
    return (s[:160] + "...") if len(s) > 160 else s


def _evaluate(name, seed, build_and_train):
    """Run one model's ``build_and_train() -> (metrics, params, seconds)`` with
    a fresh session, fixed seed, and error isolation."""
    reset_session()
    set_seeds(seed)
    try:
        metrics, params, secs = build_and_train()
        return ModelResult(name, "ok", metrics, params, secs)
    except SkipModel as exc:
        return ModelResult(name, "skipped", note=str(exc))
    except Exception as exc:  # one bad model must never abort the sweep
        return ModelResult(name, "failed", note=_short(exc))


def _print_progress(track, r):
    bits = " ".join("%s=%s" % (k, _fmt(v)) for k, v in r.metrics.items())
    extra = "" if r.train_seconds is None else " (%.1fs)" % r.train_seconds
    note = "" if not r.note else " :: %s" % r.note
    print("[%s] %-12s %-7s %s%s%s" % (track, r.model, r.status, bits, extra, note))


def _finish(results, primary_metric, track, dataset, out_dir, title):
    results = sort_results(results, primary_metric)
    print_leaderboard(results, primary_metric, title)
    csv_path, md_path = write_results(results, out_dir, track, dataset, primary_metric)
    print("wrote %s" % csv_path)
    print("wrote %s" % md_path)
    return results


# --------------------------------------------------------------------------- #
# Track runners.
# --------------------------------------------------------------------------- #
def run_single(args):
    data = load_criteo(
        source="download" if args.download else args.source,
        data_path=args.data_path, embedding_dim=args.embedding_dim, use_hash=args.hash,
        hash_buckets=args.hash_buckets,
        test_size=args.test_size, seed=args.seed, max_rows=args.max_rows,
        download_url=args.download_url,
    )
    names = select(list(SINGLE_TASK_MODELS), args.models, args.exclude)
    if args.quick:
        names = names[:2]

    loss = "binary_crossentropy" if data.task == "binary" else "mse"
    compile_metric = ["binary_crossentropy"] if data.task == "binary" else ["mse"]
    primary = "AUC" if data.task == "binary" else "RMSE"

    print("== single-task track | dataset=%s | %d models | task=%s ==" % (data.name, len(names), data.task))
    results = []
    for name in names:
        def build_and_train(name=name):
            model = SINGLE_TASK_MODELS[name](data.linear_feature_columns, data.dnn_feature_columns, data.task)
            model.compile("adam", loss, metrics=compile_metric)
            t0 = time.time()
            model.fit(data.train_input, data.train_y, batch_size=args.batch_size,
                      epochs=args.epochs, verbose=args.fit_verbose, validation_split=args.val_split)
            secs = time.time() - t0
            pred = model.predict(data.test_input, batch_size=args.batch_size)
            return compute_metrics(data.task, data.test_y, pred), model.count_params(), secs

        r = _evaluate(name, args.seed, build_and_train)
        _print_progress("single", r)
        results.append(r)

    return _finish(results, primary, "single", data.name, args.output_dir,
                   "Single-task CTR leaderboard (%s)" % data.name)


def run_multitask(args):
    data = load_census(data_path=args.data_path, embedding_dim=args.embedding_dim,
                       test_size=args.test_size, seed=args.seed)
    names = select(list(MULTITASK_MODELS), args.models, args.exclude)
    if args.quick:
        names = names[:2]

    n_tasks = len(data.task_types)
    print("== multitask track | dataset=%s | %d models | tasks=%s ==" % (data.name, len(names), data.task_names))
    results = []
    for name in names:
        def build_and_train(name=name):
            model = MULTITASK_MODELS[name](data.dnn_feature_columns, data.task_types, data.task_names)
            # No compile-time metrics: a flat list is applied to every output and
            # collides on names; we compute AUC/LogLoss from predictions anyway.
            model.compile("adam", loss=["binary_crossentropy"] * n_tasks)
            t0 = time.time()
            model.fit(data.train_input, data.train_y, batch_size=args.batch_size,
                      epochs=args.epochs, verbose=args.fit_verbose, validation_split=args.val_split)
            secs = time.time() - t0
            preds = model.predict(data.test_input, batch_size=args.batch_size)
            if not isinstance(preds, (list, tuple)):
                preds = [preds]
            merged, aucs = {}, []
            for i, tname in enumerate(data.task_names):
                m = compute_metrics("binary", data.test_y[i], preds[i])
                for k, v in m.items():
                    merged["%s_%s" % (tname, k)] = v
                if m["AUC"] == m["AUC"]:  # not NaN
                    aucs.append(m["AUC"])
            merged["mean_AUC"] = round(sum(aucs) / len(aucs), 4) if aucs else float("nan")
            return merged, model.count_params(), secs

        r = _evaluate(name, args.seed, build_and_train)
        _print_progress("multitask", r)
        results.append(r)

    return _finish(results, "mean_AUC", "multitask", data.name, args.output_dir,
                   "Multitask leaderboard (%s)" % data.name)


def run_sequence(args):
    n_samples = 400 if args.quick else args.n_samples
    din_data, dsin_data = make_sequence_datasets(
        source=args.seq_source, n_samples=n_samples, max_hist=args.sess_max_count * args.sess_len,
        sess_max_count=args.sess_max_count, sess_len=args.sess_len, embedding_dim=args.embedding_dim,
        test_size=args.test_size, seed=args.seed, data_path=args.data_path,
    )
    dataset_name = din_data.name
    names = select(list(SEQUENCE_MODELS), args.models, args.exclude)
    if args.quick:
        names = names[:2]

    print("== sequence track | dataset=%s | %d models ==" % (dataset_name, len(names)))
    results = []
    for name in names:
        spec = SEQUENCE_MODELS[name]
        data = din_data if spec["view"] == "din" else dsin_data

        def build_and_train(spec=spec, data=data):
            model = spec["build"](data, "binary")
            model.compile("adam", "binary_crossentropy", metrics=["binary_crossentropy"])
            t0 = time.time()
            model.fit(data.train_input, data.train_y, batch_size=args.batch_size,
                      epochs=args.epochs, verbose=args.fit_verbose, validation_split=args.val_split)
            secs = time.time() - t0
            pred = model.predict(data.test_input, batch_size=args.batch_size)
            return compute_metrics("binary", data.test_y, pred), model.count_params(), secs

        r = _evaluate(name, args.seed, build_and_train)
        _print_progress("sequence", r)
        results.append(r)

    return _finish(results, "AUC", "sequence", dataset_name, args.output_dir,
                   "Sequence leaderboard (%s)" % dataset_name)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def _csv_list(value):
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def build_parser():
    from .datasets.criteo import DEFAULT_DAC_URL

    p = argparse.ArgumentParser(description="DeepCTR model evaluation / benchmark pipeline")
    p.add_argument("--track", choices=["single", "multitask", "sequence", "all"], default="single")
    # data sources
    p.add_argument("--source", choices=["bundled", "download"], default="bundled",
                   help="single-task: bundled sample (default) or download Criteo DAC")
    p.add_argument("--download", action="store_true", help="shortcut for --source download")
    p.add_argument("--download-url", default=DEFAULT_DAC_URL)
    p.add_argument("--data-path", default=None, help="path to a full Criteo / Census / MovieLens file")
    p.add_argument("--hash", action="store_true", help="single-task: hash sparse features (no LabelEncoder)")
    p.add_argument("--hash-buckets", type=int, default=1000,
                   help="single-task --hash: bucket count per sparse feature (raise for high-cardinality data)")
    p.add_argument("--max-rows", type=int, default=None, help="subsample the first N rows (large files)")
    p.add_argument("--seq-source", choices=["synthetic", "movielens"], default="synthetic")
    p.add_argument("--n-samples", type=int, default=2000, help="synthetic sequence sample count")
    p.add_argument("--sess-max-count", type=int, default=3, help="DSIN sessions per user")
    p.add_argument("--sess-len", type=int, default=4, help="DSIN session length")
    # model selection
    p.add_argument("--models", type=_csv_list, default=None, help="comma-separated subset to run")
    p.add_argument("--exclude", type=_csv_list, default=None, help="comma-separated models to skip")
    # training
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--embedding-dim", type=int, default=8)
    p.add_argument("--seed", type=int, default=2020)
    p.add_argument("--fit-verbose", type=int, default=0, help="passed to model.fit(verbose=...)")
    p.add_argument("--quick", action="store_true", help="fast self-test: 1 epoch, 2 models/track, small data")
    p.add_argument("--output-dir", default=DEFAULT_RESULTS_DIR)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.quick:
        args.epochs = 1

    runners = {"single": run_single, "multitask": run_multitask, "sequence": run_sequence}
    if args.track == "all":
        for track in ["single", "multitask", "sequence"]:
            runners[track](args)
    else:
        runners[args.track](args)


if __name__ == "__main__":
    main()
