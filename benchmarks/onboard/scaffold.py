"""Scaffold a new model and auto-wire every registration point.

Eliminates the error-prone 6-point manual checklist that left OneTrans
half-integrated. Generates model/test/layer skeletons from templates and edits
all the __init__ / custom_objects / registry files idempotently.

  scaffold <Name> --category single|sequence|multitask [--with-layer] [--wire-only]
"""
from __future__ import absolute_import, division, print_function

import os
import re

from . import REPO_ROOT
from ._util import (
    CATEGORY_INIT, CATEGORY_PKG_DIR, CATEGORY_REGISTRY, LAYERS_INIT, MODELS_INIT, REGISTRY_PY,
    TESTS_MODELS_DIR, add_dict_entry, add_to_all, add_to_import_line, add_to_paren_import,
    ensure_line, layers_custom_objects, read_text, write_text,
)

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
TESTS_LAYERS_DIR = os.path.join(REPO_ROOT, "tests", "layers")


def _render(template_name, mapping):
    text = read_text(os.path.join(TEMPLATE_DIR, template_name))
    for k, v in mapping.items():
        text = text.replace("{{%s}}" % k, v)
    return text


def _log(changed, msg):
    print(("  [+] " if changed else "  [=] ") + msg)


# --------------------------------------------------------------------------- #
# Individual wiring steps
# --------------------------------------------------------------------------- #
def _wire_package_exports(name, category):
    """Make ``from deepctr.models import <Name>`` work."""
    lower = name.lower()
    init = CATEGORY_INIT[category]
    if category == "single":
        changed = ensure_line(init, "from .%s import %s" % (lower, name),
                              anchor="__all__ = [", after=False)
        _log(changed, "deepctr/models/__init__.py: import %s" % name)
    else:
        # sub-package __init__ holds the definition import …
        changed = ensure_line(init, "from .%s import %s" % (lower, name))
        _log(changed, "deepctr/models/%s/__init__.py: import %s" % (category, name))
        # … and the top-level package re-exports via the aggregate line.
        changed = add_to_import_line(MODELS_INIT, category, name)
        _log(changed, "deepctr/models/__init__.py: re-export %s from .%s" % (name, category))
    changed = add_to_all(MODELS_INIT, name)
    _log(changed, "deepctr/models/__init__.py: __all__ += %s" % name)


def _wire_custom_objects(name, layer_symbols):
    """Register layer classes in deepctr/layers/custom_objects."""
    for sym in layer_symbols:
        changed = add_dict_entry(LAYERS_INIT, "custom_objects", sym, sym)
        _log(changed, "deepctr/layers/__init__.py: custom_objects['%s']" % sym)


def _detect_model_layer_symbols(name, category):
    """For --wire-only: find layer symbols this model needs that are already
    imported in layers/__init__ but missing from custom_objects.

    Matches by the layer module sharing the model stem (e.g. ``onetrans``).
    """
    lower = name.lower()
    imported, registered = layers_custom_objects()
    text = read_text(LAYERS_INIT)
    # names imported from `.<lower>` module in layers/__init__
    m = re.search(r"^from\s+\.%s\s+import\s+(.+)$" % re.escape(lower), text, re.MULTILINE)
    syms = []
    if m:
        for tok in m.group(1).split(","):
            tok = tok.strip()
            if " as " in tok:
                tok = tok.split(" as ")[-1].strip()
            if tok and tok not in registered:
                syms.append(tok)
    return syms


def _create_layer_file(name):
    lower = name.lower()
    path = os.path.join(REPO_ROOT, "deepctr", "layers", lower + ".py")
    if os.path.exists(path):
        _log(False, "deepctr/layers/%s.py exists" % lower)
    else:
        write_text(path, _render("layer.py.tmpl", {"NAME": name}))
        _log(True, "deepctr/layers/%s.py created" % lower)
    changed = ensure_line(LAYERS_INIT, "from .%s import %sLayer" % (lower, name),
                          anchor="custom_objects = {", after=False)
    _log(changed, "deepctr/layers/__init__.py: import %sLayer" % name)
    return [name + "Layer"]


def _create_layer_test(name):
    path = os.path.join(TESTS_LAYERS_DIR, name + "_correctness_test.py")
    if os.path.exists(path):
        _log(False, "tests/layers/%s_correctness_test.py exists" % name)
        return
    mapping = {"NAME": name, "NAME_LOWER": name.lower()}
    write_text(path, _render("test_layer_correctness.py.tmpl", mapping))
    _log(True, "tests/layers/%s_correctness_test.py created" % name)


def _create_model_file(name, category, mapping):
    lower = name.lower()
    pkg_dir = CATEGORY_PKG_DIR[category]
    path = os.path.join(pkg_dir, lower + ".py")
    if os.path.exists(path):
        _log(False, "model file %s exists (skipped)" % os.path.relpath(path, REPO_ROOT))
        return
    template = {"single": "model_single.py.tmpl",
                "sequence": "model_sequence.py.tmpl",
                "multitask": "model_multitask.py.tmpl"}[category]
    write_text(path, _render(template, mapping))
    _log(True, "%s created" % os.path.relpath(path, REPO_ROOT))


def _create_test_file(name, category):
    path = os.path.join(TESTS_MODELS_DIR, name + "_test.py")
    if os.path.exists(path):
        _log(False, "tests/models/%s_test.py exists" % name)
        return
    template = {"single": "test_single.py.tmpl",
                "sequence": "test_sequence.py.tmpl",
                "multitask": "test_multitask.py.tmpl"}[category]
    write_text(path, _render(template, {"NAME": name}))
    _log(True, "tests/models/%s_test.py created" % name)


def _wire_registry(name, category):
    # 1) import the model symbol into benchmarks/registry.py
    changed = add_to_paren_import(REGISTRY_PY, name, "deepctr.models")
    _log(changed, "benchmarks/registry.py: import %s" % name)

    # 2) add the builder entry
    dict_name = CATEGORY_REGISTRY[category]
    if category == "single":
        expr = "lambda lin, dnn, task: %s(lin, dnn, task=task)" % name
        changed = add_dict_entry(REGISTRY_PY, dict_name, name, expr)
        _log(changed, "benchmarks/registry.py: %s['%s']" % (dict_name, name))
    elif category == "multitask":
        expr = "lambda dnn, types, names: %s(dnn, task_types=types, task_names=names)" % name
        changed = add_dict_entry(REGISTRY_PY, dict_name, name, expr)
        _log(changed, "benchmarks/registry.py: %s['%s']" % (dict_name, name))
    else:  # sequence: needs a builder function + dict entry referencing it
        _wire_sequence_builder(name)


def _wire_sequence_builder(name):
    lower = name.lower()
    text = read_text(REGISTRY_PY)
    builder = "_build_%s" % lower
    if ("def %s(" % builder) not in text:
        fn = (
            "\ndef %s(data, task):\n"
            "    # Transformer-style sequence model: pick head count that divides the\n"
            "    # behavior embedding dim (see _build_bst / _build_dsin for the pattern).\n"
            "    emb = _behavior_embedding_dim(data)\n"
            "    head = _largest_divisor_leq(emb, 4)\n"
            "    return %s(data.feature_columns, data.behavior_feature_list,\n"
            "             %satt_head_num=head, task=task)\n\n\n"
            % (builder, name, " " * len(name))
        )
        anchor = "SEQUENCE_MODELS = {"
        idx = text.index(anchor)
        text = text[:idx] + fn.lstrip("\n") + text[idx:]
        write_text(REGISTRY_PY, text)
        _log(True, "benchmarks/registry.py: %s()" % builder)
    else:
        _log(False, "benchmarks/registry.py: %s() exists" % builder)
    expr = '{"view": "din", "build": %s}' % builder
    changed = add_dict_entry(REGISTRY_PY, "SEQUENCE_MODELS", name, expr)
    _log(changed, "benchmarks/registry.py: SEQUENCE_MODELS['%s']" % name)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run(args):
    name = args.name
    category = args.category
    paper = args.paper_title or "TODO: add paper citation."
    mapping = {"NAME": name, "NAME_LOWER": name.lower(), "CATEGORY": category, "PAPER": paper}

    print("Scaffolding %s (%s)%s" % (name, category, " [wire-only]" if args.wire_only else ""))

    layer_symbols = []
    if args.wire_only:
        layer_symbols = _detect_model_layer_symbols(name, category)
    else:
        _create_model_file(name, category, mapping)
        if args.with_layer:
            layer_symbols = _create_layer_file(name)
            _create_layer_test(name)

    _wire_package_exports(name, category)
    if layer_symbols:
        _wire_custom_objects(name, layer_symbols)
    _create_test_file(name, category)
    _wire_registry(name, category)

    print("Done. Run:  python -m benchmarks.onboard audit --name %s" % name)
    if not args.wire_only:
        print("Next:  implement the TODO core, then `verify` and `docs`.")
    return 0
