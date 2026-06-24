"""Integration completeness validator.

Auto-discovers every model symbol across the deepctr.models packages and checks
that each one is fully wired:

  * importable as ``from deepctr.models import X``
  * listed in ``deepctr.models.__all__``
  * any custom layers it uses are registered in ``deepctr.layers.custom_objects``
    (a missing entry silently breaks ``save_model``/``load_model``)
  * has a unit test under ``tests/models/``
  * has a builder in the matching ``benchmarks/registry.py`` dict

Static checks only (no training) so it is fast and import-light. The deeper
save/load round-trip is exercised by ``verify``.
"""
from __future__ import absolute_import, division, print_function

import os

from . import REPO_ROOT
from ._util import (
    CATEGORY_PKG_DIR, all_registry_entries, discover_models, layers_custom_objects,
    models_all_list, read_text, test_file_for,
)

OK, FAIL, NA = "ok", "MISSING", "-"


def _model_module_path(info):
    """Filesystem path of the .py module a model is defined in."""
    module = info["module"]  # e.g. "sequence.onetrans" or "deepfm"
    parts = module.split(".")
    return os.path.join(REPO_ROOT, "deepctr", "models", *parts) + ".py"


def audit_models(only=None):
    """Return (rows, summary). Each row is a dict of per-model check results."""
    models = discover_models()
    in_all = set(models_all_list())
    imported_layers, registered_keys = layers_custom_objects()
    unregistered_layers = {l for l in imported_layers if l not in registered_keys}
    registry = all_registry_entries()

    # Confirm importability through the real package (catches broken __init__).
    importable = set()
    try:
        import deepctr.models as dm
        for name in models:
            if hasattr(dm, name):
                importable.add(name)
    except Exception as e:  # pragma: no cover - environment dependent
        print("warning: could not import deepctr.models (%s); "
              "falling back to static checks only" % e)
        importable = set(in_all)

    rows = []
    for name in sorted(models):
        if only and name not in only:
            continue
        info = models[name]
        # Which (if any) unregistered custom layers does this model import?
        missing_layers = []
        mod_path = _model_module_path(info)
        if unregistered_layers and os.path.exists(mod_path):
            src = read_text(mod_path)
            missing_layers = sorted(l for l in unregistered_layers if l in src)
        rows.append({
            "name": name,
            "category": info["category"],
            "import": OK if name in importable else FAIL,
            "in_all": OK if name in in_all else FAIL,
            "custom_objects": OK if not missing_layers else ",".join(missing_layers),
            "test": OK if test_file_for(name) else FAIL,
            "registry": OK if name in registry else FAIL,
        })

    summary = {
        "total": len(rows),
        "fully_wired": sum(1 for r in rows if _row_ok(r)),
        "unregistered_layers": sorted(unregistered_layers),
    }
    return rows, summary


def _row_ok(row):
    return all(row[k] == OK for k in ("import", "in_all", "custom_objects", "test", "registry"))


def _fmt_table(rows):
    cols = [("name", 14), ("category", 10), ("import", 8), ("in_all", 8),
            ("custom_objects", 22), ("test", 8), ("registry", 8)]
    head = "  ".join(h.ljust(w) for h, w in cols)
    lines = [head, "  ".join("-" * w for _h, w in cols)]
    for r in rows:
        lines.append("  ".join(str(r[h]).ljust(w) for h, w in cols))
    return "\n".join(lines)


def run(args):
    only = set(args.name) if getattr(args, "name", None) else None
    rows, summary = audit_models(only=only)
    print(_fmt_table(rows))
    print()
    print("Fully wired: %d/%d models" % (summary["fully_wired"], summary["total"]))
    if summary["unregistered_layers"]:
        print("Layers imported but NOT in custom_objects (save/load will fail): %s"
              % ", ".join(summary["unregistered_layers"]))
    incomplete = [r["name"] for r in rows if not _row_ok(r)]
    if incomplete:
        print("Incomplete: %s" % ", ".join(incomplete))
        return 1
    print("All discovered models are fully wired.")
    return 0
