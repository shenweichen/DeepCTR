"""Shared helpers for the onboarding CLI: repo paths, model-source discovery,
and idempotent anchor-based file editing used by scaffold/docs.
"""
from __future__ import absolute_import, division, print_function

import os
import re

from . import REPO_ROOT

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
MODELS_INIT = os.path.join(REPO_ROOT, "deepctr", "models", "__init__.py")
SEQUENCE_INIT = os.path.join(REPO_ROOT, "deepctr", "models", "sequence", "__init__.py")
MULTITASK_INIT = os.path.join(REPO_ROOT, "deepctr", "models", "multitask", "__init__.py")
LAYERS_INIT = os.path.join(REPO_ROOT, "deepctr", "layers", "__init__.py")
REGISTRY_PY = os.path.join(REPO_ROOT, "benchmarks", "registry.py")
TESTS_MODELS_DIR = os.path.join(REPO_ROOT, "tests", "models")

CATEGORY_INIT = {
    "single": MODELS_INIT,
    "sequence": SEQUENCE_INIT,
    "multitask": MULTITASK_INIT,
}
CATEGORY_PKG_DIR = {
    "single": os.path.join(REPO_ROOT, "deepctr", "models"),
    "sequence": os.path.join(REPO_ROOT, "deepctr", "models", "sequence"),
    "multitask": os.path.join(REPO_ROOT, "deepctr", "models", "multitask"),
}
CATEGORY_REGISTRY = {
    "single": "SINGLE_TASK_MODELS",
    "sequence": "SEQUENCE_MODELS",
    "multitask": "MULTITASK_MODELS",
}


# --------------------------------------------------------------------------- #
# Plain file IO
# --------------------------------------------------------------------------- #
def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# --------------------------------------------------------------------------- #
# Model-source discovery
# --------------------------------------------------------------------------- #
_IMPORT_RE = re.compile(r"^from\s+\.(\S+)\s+import\s+(.+)$", re.MULTILINE)


def _parse_imports(init_path):
    """Return list of (module, [names]) from ``from .module import a, b`` lines."""
    out = []
    if not os.path.exists(init_path):
        return out
    for mod, names in _IMPORT_RE.findall(read_text(init_path)):
        clean = []
        for tok in names.split(","):
            tok = tok.strip()
            if " as " in tok:
                tok = tok.split(" as ")[-1].strip()
            if tok:
                clean.append(tok)
        out.append((mod, clean))
    return out


def discover_models():
    """Discover every model symbol across the three model __init__ files.

    Returns a dict: name -> {"category", "module", "init"}.  ``module`` is the
    dotted submodule the model is defined in (e.g. ``sequence.onetrans``).
    Captures models that exist in a sub-package even if they are NOT yet
    re-exported from the top-level ``deepctr.models`` namespace (e.g. OneTrans).
    """
    models = {}

    for mod, names in _parse_imports(SEQUENCE_INIT):
        for n in names:
            models[n] = {"category": "sequence", "module": "sequence." + mod, "init": SEQUENCE_INIT}
    for mod, names in _parse_imports(MULTITASK_INIT):
        for n in names:
            models[n] = {"category": "multitask", "module": "multitask." + mod, "init": MULTITASK_INIT}

    # Top-level: direct `from .mod import Name` are single-task; the aggregate
    # `from .multitask/.sequence import ...` lines are skipped (already covered).
    for mod, names in _parse_imports(MODELS_INIT):
        if mod in ("multitask", "sequence"):
            continue
        for n in names:
            models.setdefault(n, {"category": "single", "module": mod, "init": MODELS_INIT})

    return models


def models_all_list():
    """Return the names currently in ``deepctr.models.__all__`` (textually)."""
    text = read_text(MODELS_INIT)
    m = re.search(r"__all__\s*=\s*\[(.*?)\]", text, re.DOTALL)
    if not m:
        return []
    return re.findall(r"""['"]([A-Za-z_][\w]*)['"]""", m.group(1))


def layers_custom_objects():
    """Return (imported_layer_names, registered_keys) from deepctr/layers/__init__.py.

    ``imported_layer_names`` are CapWords symbols imported into the layers package;
    ``registered_keys`` are the string keys present in the ``custom_objects`` dict.
    """
    text = read_text(LAYERS_INIT)
    imported = set()
    for _mod, names in _parse_imports(LAYERS_INIT):
        for n in names:
            if n[:1].isupper():  # layer classes are CapWords
                imported.add(n)
    co = re.search(r"custom_objects\s*=\s*\{(.*?)\n\s*\}", text, re.DOTALL)
    registered = set(re.findall(r"""['"]([\w]+)['"]\s*:""", co.group(1))) if co else set()
    return imported, registered


def registry_entries(category):
    """Return the set of model names registered in the given registry dict."""
    text = read_text(REGISTRY_PY)
    dict_name = CATEGORY_REGISTRY[category]
    m = re.search(dict_name + r"\s*=\s*\{(.*?)\n\}", text, re.DOTALL)
    if not m:
        return set()
    return set(re.findall(r"""['"]([\w]+)['"]\s*:""", m.group(1)))


def all_registry_entries():
    out = {}
    for cat in CATEGORY_REGISTRY:
        for name in registry_entries(cat):
            out[name] = cat
    return out


def test_file_for(name):
    """Path to the per-model test file if it exists, else None.

    Also treats the shared multitask test (MTL_test.py) as covering a model when
    the model name appears inside it.
    """
    direct = os.path.join(TESTS_MODELS_DIR, name + "_test.py")
    if os.path.exists(direct):
        return direct
    # fall back: any test file that references the model symbol
    if os.path.isdir(TESTS_MODELS_DIR):
        for fn in os.listdir(TESTS_MODELS_DIR):
            if fn.endswith("_test.py"):
                p = os.path.join(TESTS_MODELS_DIR, fn)
                if re.search(r"\b" + re.escape(name) + r"\b", read_text(p)):
                    return p
    return None


# --------------------------------------------------------------------------- #
# Idempotent anchor-based editing
# --------------------------------------------------------------------------- #
def ensure_line(path, line, anchor=None, after=True):
    """Insert ``line`` into ``path`` if not already present (idempotent).

    If ``anchor`` (a substring) is given, insert immediately after (or before)
    the line containing it; otherwise append to end of file. Returns True if the
    file was modified.
    """
    text = read_text(path)
    if line in text:
        return False
    lines = text.splitlines(keepends=True)
    if anchor is None:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(line + "\n")
    else:
        idx = next((i for i, l in enumerate(lines) if anchor in l), None)
        if idx is None:
            raise ValueError("anchor %r not found in %s" % (anchor, path))
        insert_at = idx + 1 if after else idx
        lines.insert(insert_at, line + "\n")
    write_text(path, "".join(lines))
    return True


def add_to_import_line(path, module, name):
    """Append ``name`` to an existing ``from .<module> import a, b`` line. Idempotent."""
    text = read_text(path)
    pat = re.compile(r"^(from\s+\.%s\s+import\s+)(.+)$" % re.escape(module), re.MULTILINE)
    m = pat.search(text)
    if not m:
        raise ValueError("import line for module %r not found in %s" % (module, path))
    names = m.group(2)
    if re.search(r"\b%s\b" % re.escape(name), names):
        return False
    new = m.group(1) + names.rstrip() + ", " + name
    write_text(path, text[:m.start()] + new + text[m.end():])
    return True


def add_to_paren_import(path, name, module="deepctr.models"):
    """Add ``name`` into a parenthesized ``from <module> import (a, b, ...)`` block. Idempotent."""
    text = read_text(path)
    pat = re.compile(r"(from\s+%s\s+import\s+\()(.*?)(\))" % re.escape(module), re.DOTALL)
    m = pat.search(text)
    if not m:
        raise ValueError("parenthesized import from %r not found in %s" % (module, path))
    body = m.group(2)
    if re.search(r"\b%s\b" % re.escape(name), body):
        return False
    new_body = body.rstrip()
    if not new_body.endswith(","):
        new_body += ","
    new_body += " " + name
    write_text(path, text[:m.start(2)] + new_body + text[m.end(2):])
    return True


def add_to_all(path, name):
    """Add ``name`` to the module's ``__all__`` list if missing. Idempotent."""
    text = read_text(path)
    m = re.search(r"(__all__\s*=\s*\[)(.*?)(\])", text, re.DOTALL)
    if not m:
        raise ValueError("no __all__ found in %s" % path)
    body = m.group(2)
    if re.search(r"""['"]%s['"]""" % re.escape(name), body):
        return False
    new_body = body.rstrip()
    if not new_body.endswith(","):
        new_body += ","
    new_body += ' "%s"' % name
    write_text(path, text[:m.start(2)] + new_body + text[m.end(2):])
    return True


def add_dict_entry(path, dict_name, key, value_expr, dict_close=r"\n\s*\}"):
    """Add ``'key': value_expr`` before the closing brace of ``dict_name``.

    Idempotent on the key. Returns True if modified.
    """
    text = read_text(path)
    pat = re.compile(dict_name + r"\s*=\s*\{(.*?)(\n[ \t]*\})", re.DOTALL)
    m = pat.search(text)
    if not m:
        raise ValueError("dict %s not found in %s" % (dict_name, path))
    body = m.group(1)
    if re.search(r"""['"]%s['"]\s*:""" % re.escape(key), body):
        return False
    indent = "                  "  # match deepctr/layers custom_objects indent
    inner_indent = re.match(r"\n([ \t]*)", body)
    if inner_indent:
        indent = inner_indent.group(1)
    new_body = body.rstrip()
    if not new_body.endswith(",") and new_body.strip():
        new_body += ","
    new_body += "\n%s'%s': %s," % (indent, key, value_expr)
    write_text(path, text[:m.start(1)] + new_body + text[m.end(1):])
    return True
