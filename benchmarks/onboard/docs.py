"""Stage 4 - update documentation for a model (idempotent, anchor-based).

Touches:
  - README.md                         : add a row to the model table
  - docs/source/Features.md           : add a ### subsection in the right section
  - docs/source/deepctr.models.rst    : add a toctree entry
  - docs/source/deepctr.models.<...>.rst : create the autodoc stub
  - docs/source/History.md            : add a dated changelog bullet
  - benchmarks/RESULTS.md             : record the latest verify leaderboard line
"""
from __future__ import absolute_import, division, print_function

import os
import re

from . import REPO_ROOT
from ._util import discover_models, read_text, write_text

README = os.path.join(REPO_ROOT, "README.md")
FEATURES = os.path.join(REPO_ROOT, "docs", "source", "Features.md")
MODELS_RST = os.path.join(REPO_ROOT, "docs", "source", "deepctr.models.rst")
DOCS_SRC = os.path.join(REPO_ROOT, "docs", "source")
HISTORY = os.path.join(REPO_ROOT, "docs", "source", "History.md")
RESULTS = os.path.join(REPO_ROOT, "benchmarks", "RESULTS.md")

# where a ### subsection goes, by category: insert before this Features.md heading
FEATURES_ANCHOR = {
    "single": "## Sequence Models",
    "sequence": "## MultiTask Models",
    "multitask": "## Layers",
}
# rst module dotted path, by category
RST_MODULE = {
    "single": "deepctr.models.%s",
    "sequence": "deepctr.models.sequence.%s",
    "multitask": "deepctr.models.multitask.%s",
}


def _log(changed, msg):
    print(("  [+] " if changed else "  [=] ") + msg)


def _candidate_meta(name):
    try:
        from .discover import load_candidates
        for c in load_candidates():
            if c.get("name") == name:
                return c
    except Exception:
        pass
    return {}


# --------------------------------------------------------------------------- #
def _update_readme_table(name, venue, year, title, url):
    text = read_text(README)
    if re.search(r"\|\s*%s\s*\|" % re.escape(name), text):
        _log(False, "README.md: table row")
        return
    lines = text.splitlines(keepends=True)
    # find the model table header
    hdr = next((i for i, l in enumerate(lines)
                if "Model" in l and "Paper" in l and l.lstrip().startswith("|")), None)
    if hdr is None:
        _log(False, "README.md: no model table found (skipped)")
        return
    # last consecutive table row after the header/separator
    i = hdr + 1
    last = i
    while i < len(lines) and lines[i].lstrip().startswith("|"):
        last = i
        i += 1
    cite = "[%s %s][%s](%s)" % (venue or "", year or "", title or name, url or "")
    row = "|   %s                   | %s   |\n" % (name, cite)
    lines.insert(last + 1, row)
    write_text(README, "".join(lines))
    _log(True, "README.md: table row")


def _update_features(name, category, one_liner, title, url):
    text = read_text(FEATURES)
    if re.search(r"^###\s+%s\b" % re.escape(name), text, re.MULTILINE):
        _log(False, "Features.md: ### %s" % name)
        return
    anchor = FEATURES_ANCHOR[category]
    block = ("### %s\n\n%s\n\n[%s](%s)\n\n"
             % (name, one_liner or "TODO: description.", title or name, url or ""))
    if anchor in text:
        text = text.replace(anchor, block + anchor, 1)
    else:
        text = text.rstrip() + "\n\n" + block
    write_text(FEATURES, text)
    _log(True, "Features.md: ### %s" % name)


def _update_rst(name, category):
    lower = name.lower()
    module = RST_MODULE[category] % lower
    stub_path = os.path.join(DOCS_SRC, module + ".rst")
    if os.path.exists(stub_path):
        _log(False, "%s.rst exists" % module)
    else:
        title = "%s module" % module
        underline = "=" * len(title)
        stub = ("%s\n%s\n\n.. automodule:: %s\n    :members:\n"
                "    :no-undoc-members:\n    :no-show-inheritance:\n"
                % (title, underline, module))
        write_text(stub_path, stub)
        _log(True, "%s.rst created" % module)

    # add to toctree in deepctr.models.rst
    text = read_text(MODELS_RST)
    entry = "   %s\n" % module
    if entry in text:
        _log(False, "deepctr.models.rst: toctree %s" % module)
        return
    lines = text.splitlines(keepends=True)
    # insert after the last existing toctree model entry
    last = next((i for i in range(len(lines) - 1, -1, -1)
                 if lines[i].lstrip().startswith("deepctr.models.")), None)
    if last is None:
        _log(False, "deepctr.models.rst: no toctree found (skipped)")
        return
    lines.insert(last + 1, entry)
    write_text(MODELS_RST, "".join(lines))
    _log(True, "deepctr.models.rst: toctree %s" % module)


def _update_history(name, date_str, anchor_slug):
    text = read_text(HISTORY)
    if re.search(r"Add \[%s\]" % re.escape(name), text):
        _log(False, "History.md: changelog bullet")
        return
    lines = text.splitlines(keepends=True)
    hidx = next((i for i, l in enumerate(lines) if l.strip() == "# History"), None)
    bullet = ("- %s : Add [%s](./Features.html#%s) model via the onboarding pipeline.\n"
              % (date_str, name, anchor_slug))
    if hidx is None:
        lines.insert(0, bullet)
    else:
        lines.insert(hidx + 1, bullet)
    write_text(HISTORY, "".join(lines))
    _log(True, "History.md: changelog bullet")


def _update_results(name):
    """Append the latest verify leaderboard line into a managed RESULTS section."""
    from .verify import REPORTS_DIR
    report = os.path.join(REPORTS_DIR, name + ".md")
    if not os.path.exists(report):
        _log(False, "RESULTS.md: no verify report yet (run verify first)")
        return
    marker = "<!-- onboard:results -->"
    text = read_text(RESULTS) if os.path.exists(RESULTS) else "# Benchmark results\n"
    if marker not in text:
        text = text.rstrip() + ("\n\n## Onboarded via pipeline %s\n\n"
                                "Auto-recorded verify outcomes for models added through "
                                "`benchmarks.onboard`.\n\n" % marker)
    line = "- **%s**: see `benchmarks/onboard/reports/%s.md`\n" % (name, name)
    if line in text:
        _log(False, "RESULTS.md: entry for %s" % name)
        return
    text = text.rstrip() + "\n" + line
    write_text(RESULTS, text)
    _log(True, "RESULTS.md: entry for %s" % name)


def run(args):
    name = args.name
    category = getattr(args, "category", None) or discover_models().get(name, {}).get("category")
    if not category:
        raise SystemExit("unknown model %r; pass --category" % name)

    meta = _candidate_meta(name)
    title = args.paper_title or meta.get("paper_title", "")
    url = args.paper_url or meta.get("paper_url", "")
    one_liner = getattr(args, "one_liner", "") or meta.get("one_liner", "")
    venue = meta.get("venue", "")
    year = meta.get("year", "")
    # anchor slug Sphinx generates from "### Name"
    anchor_slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")

    print("Updating docs for %s (%s)" % (name, category))
    _update_readme_table(name, venue, year, title, url)
    _update_features(name, category, one_liner, title, url)
    _update_rst(name, category)
    _update_history(name, "06/24/2026", anchor_slug)
    _update_results(name)
    print("Docs updated.")
    return 0
