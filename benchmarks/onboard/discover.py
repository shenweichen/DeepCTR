"""Stage 1 - discover candidate models.

Maintains a version-controlled knowledge base (``candidates.json``) of recent
industry/academic CTR models, each with a fixed schema (paper, venue, category,
key idea, reference impl, reported metric, integration difficulty, status).

  discover --list                list candidates (optionally filtered)
  discover --refresh             print a research prompt to refresh the KB via
                                 web search / the deep-research skill

Discovery cannot fully auto-select models (that needs judgement), so the CLI
automates the structured/ranked part and marks which candidates are already
implemented by cross-referencing the live deepctr.models packages.
"""
from __future__ import absolute_import, division, print_function

import json
import os

from ._util import discover_models

CANDIDATES_PATH = os.path.join(os.path.dirname(__file__), "candidates.json")

DIFFICULTY_RANK = {"easy": 0, "medium": 1, "hard": 2}

RESEARCH_PROMPT = """\
Refresh benchmarks/onboard/candidates.json with recent CTR / recommendation models.

Search these sources for models published in the last ~3 years that fit DeepCTR's
feature-column interface (SparseFeat / DenseFeat / VarLenSparseFeat):
  - arXiv (cs.IR / cs.LG), Papers-with-Code CTR leaderboards
  - KDD / RecSys / CIKM / SIGIR / WWW / AAAI proceedings
  - FuxiCTR / BARS benchmark model zoo (github.com/reczoo/FuxiCTR)

For each new model emit one JSON object with this schema and append it under
"candidates" (skip ones already present; keep existing "status"):
  name, paper_title, paper_url, year, venue, category(single|sequence|multitask),
  one_liner, ref_impl_url, paper_metric({dataset, AUC}), difficulty, status="candidate".

Rank by: interface fit > reported AUC on Criteo_x1/Avazu > citation count, and
prefer models NOT already in deepctr.models.
"""


def load_candidates():
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        return json.load(f).get("candidates", [])


def _implemented_names():
    return set(discover_models())


def run(args):
    if getattr(args, "refresh", False):
        print(RESEARCH_PROMPT)
        print("After editing candidates.json, run: python -m benchmarks.onboard discover --list")
        return 0

    candidates = load_candidates()
    implemented = _implemented_names()
    # reconcile status against the live packages
    for c in candidates:
        if c["name"] in implemented:
            c["status"] = "implemented"

    cat = getattr(args, "category", None)
    status = getattr(args, "status", None)
    rows = [c for c in candidates
            if (not cat or c.get("category") == cat)
            and (not status or c.get("status") == status)]
    # candidates first (not implemented), then by difficulty, then by paper AUC desc
    def sort_key(c):
        impl = c.get("status") == "implemented"
        auc = (c.get("paper_metric") or {}).get("AUC", 0) or 0
        return (impl, DIFFICULTY_RANK.get(c.get("difficulty"), 9), -auc)
    rows.sort(key=sort_key)

    cols = [("name", 12), ("category", 10), ("year", 5), ("venue", 9),
            ("difficulty", 10), ("status", 12), ("AUC", 8)]
    print("  ".join(h.ljust(w) for h, w in cols))
    print("  ".join("-" * w for _h, w in cols))
    for c in rows:
        auc = (c.get("paper_metric") or {}).get("AUC", "")
        vals = [c.get("name", ""), c.get("category", ""), str(c.get("year", "")),
                c.get("venue", ""), c.get("difficulty", ""), c.get("status", ""), str(auc)]
        print("  ".join(str(v).ljust(w) for v, (_h, w) in zip(vals, cols)))

    todo = [c["name"] for c in rows if c.get("status") == "candidate"]
    print()
    print("%d candidate(s) not yet implemented: %s" % (len(todo), ", ".join(todo) or "-"))
    print("Next: python -m benchmarks.onboard scaffold <Name> --category <cat>")
    return 0
