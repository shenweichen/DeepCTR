"""CLI entry point: ``python -m benchmarks.onboard <command> [options]``.

Commands
--------
  discover   list / refresh the candidate-model knowledge base
  scaffold   generate model+test skeletons and auto-wire all registration points
  audit      check that discovered models are fully wired
  verify     prove correctness (unit test + audit) and effectiveness (benchmark)
  docs       update README / Features / rst / History / RESULTS for a model
  onboard    run scaffold -> verify -> docs end to end
"""
from __future__ import absolute_import, division, print_function

import argparse
import sys


def build_parser():
    p = argparse.ArgumentParser(prog="python -m benchmarks.onboard",
                                description="DeepCTR model-onboarding pipeline.")
    sub = p.add_subparsers(dest="command")

    # -- discover --
    d = sub.add_parser("discover", help="list/refresh candidate models")
    d.add_argument("--list", action="store_true", help="list candidates (default)")
    d.add_argument("--refresh", action="store_true",
                   help="print a research prompt to refresh candidates.json via web search")
    d.add_argument("--category", choices=["single", "sequence", "multitask"], default=None)
    d.add_argument("--status", default=None, help="filter by status (candidate/implemented/skipped)")

    # -- scaffold --
    s = sub.add_parser("scaffold", help="generate + wire a new model")
    s.add_argument("name")
    s.add_argument("--category", choices=["single", "sequence", "multitask"], required=True)
    s.add_argument("--with-layer", action="store_true", help="also create a custom layer file")
    s.add_argument("--wire-only", action="store_true",
                   help="model already exists; only add registration + test + registry")
    s.add_argument("--paper-title", default="")
    s.add_argument("--paper-url", default="")

    # -- audit --
    a = sub.add_parser("audit", help="check model wiring completeness")
    a.add_argument("--name", nargs="*", help="restrict to these model names")

    # -- verify --
    v = sub.add_parser("verify", help="correctness + effectiveness validation")
    v.add_argument("name")
    v.add_argument("--category", choices=["single", "sequence", "multitask"], default=None)
    v.add_argument("--data-path", default=None, help="real dataset path for the benchmark")
    v.add_argument("--epochs", type=int, default=1)
    v.add_argument("--batch-size", type=int, default=1024)
    v.add_argument("--baseline", default=None,
                   help="comparison model (default: in-track baseline — "
                        "DeepFM/DIN/SharedBottom)")
    v.add_argument("--quick", action="store_true", help="bundled tiny data, smoke only")
    v.add_argument("--skip-benchmark", action="store_true", help="run correctness checks only")

    # -- docs --
    dc = sub.add_parser("docs", help="update documentation for a model")
    dc.add_argument("name")
    dc.add_argument("--category", choices=["single", "sequence", "multitask"], default=None)
    dc.add_argument("--paper-title", default="")
    dc.add_argument("--paper-url", default="")
    dc.add_argument("--one-liner", default="")

    # -- onboard --
    o = sub.add_parser("onboard", help="scaffold -> verify -> docs")
    o.add_argument("name")
    o.add_argument("--category", choices=["single", "sequence", "multitask"], required=True)
    o.add_argument("--with-layer", action="store_true")
    o.add_argument("--wire-only", action="store_true")
    o.add_argument("--quick", action="store_true")
    o.add_argument("--paper-title", default="")
    o.add_argument("--paper-url", default="")
    o.add_argument("--one-liner", default="")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.command:
        build_parser().print_help()
        return 0

    if args.command == "audit":
        from . import audit
        return audit.run(args)
    if args.command == "discover":
        from . import discover
        return discover.run(args)
    if args.command == "scaffold":
        from . import scaffold
        return scaffold.run(args)
    if args.command == "verify":
        from . import verify
        return verify.run(args)
    if args.command == "docs":
        from . import docs
        return docs.run(args)
    if args.command == "onboard":
        from . import scaffold, verify, docs
        rc = scaffold.run(args)
        if rc:
            return rc
        if not args.wire_only:
            print("\n>>> Implement the model core (marked with TODO), then re-run "
                  "verify/docs. Pausing onboard before verification.")
            return 0
        rc = verify.run(args)
        if rc:
            print("\nverify failed; fix issues before updating docs.")
            return rc
        return docs.run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
