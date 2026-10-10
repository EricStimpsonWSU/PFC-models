#!/usr/bin/env python3
"""variant_check.py — keep deliberately-forked model variants honest.

This does NOT try to deduplicate your code. It assumes forking is intentional
and only answers one question: *which functions disagree across forks, and did
that set change since you last accepted it?*

Workflow
--------
    python variant_check.py scan                 # human-readable divergence table
    python variant_check.py accept               # record today's state as expected
    python variant_check.py check                # exit 1 if divergence changed
    python variant_check.py show <module> <func> # side-by-side diff of every version

`accept` writes .variant-baseline.json — commit it. `check` is cheap enough for
a pre-commit hook:

    # .git/hooks/pre-commit
    python tools/variant_check.py check || exit 1

Comparison ignores comments, blank lines and whitespace, so reformatting a fork
does not trip the check; a changed expression does.

Stdlib only. Python 3.9+.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import re
import sys
from collections import defaultdict
from fnmatch import fnmatch
from pathlib import Path

BASELINE = ".variant-baseline.json"
SKIP_DIRS = {".git", ".venv", ".venvFFTW", "venv", "__pycache__", ".ipynb_checkpoints", ".vscode"}
_WS = re.compile(r"\s+")


def iter_sources(root: Path, globs):
    for pattern in globs:
        for p in sorted(root.rglob(pattern)):
            if not any(part in SKIP_DIRS for part in p.parts):
                yield p


def normalize(src: str) -> str:
    """Comment-, blank- and whitespace-insensitive form of a function body."""
    out = []
    for line in src.splitlines():
        line = line.split("#")[0]
        line = _WS.sub(" ", line).strip()
        if line:
            out.append(line)
    return "\n".join(out)


def functions(path: Path) -> dict:
    """Qualified function name -> source text, for one file."""
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src)
    except (SyntaxError, OSError) as exc:
        print(f"  ! skipped {path}: {exc}", file=sys.stderr)
        return {}
    found = {}

    def walk(node, prefix=""):
        for child in node.body:
            if isinstance(child, ast.ClassDef):
                walk(child, prefix + child.name + ".")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found[prefix + child.name] = ast.get_source_segment(src, child) or ""

    walk(tree)
    return found


def collect(root: Path, globs, exclude=()):
    """module filename -> variant directory -> {function: source}."""
    mods = defaultdict(dict)
    for p in iter_sources(root, globs):
        variant = p.relative_to(root).parent.as_posix() or "."
        if any(fnmatch(variant, pat) for pat in exclude):
            continue
        mods[p.name][variant] = functions(p)
    return mods


def digest(src: str) -> str:
    return hashlib.sha256(normalize(src).encode()).hexdigest()[:10]


def survey(mods):
    """(module, function) -> {hash: [variants]} for modules with >1 copy."""
    table = {}
    for module, variants in mods.items():
        if len(variants) < 2:
            continue
        names = sorted({f for fs in variants.values() for f in fs})
        for fn in names:
            groups = defaultdict(list)
            for v, fs in variants.items():
                if fn in fs:
                    groups[digest(fs[fn])].append(v)
            if groups:
                table[(module, fn)] = {h: sorted(vs) for h, vs in groups.items()}
    return table


def fingerprint(table):
    """Baseline form: version-group membership per (module, function)."""
    return {
        f"{m}::{f}": sorted(sorted(vs) for vs in groups.values())
        for (m, f), groups in table.items()
    }


def cmd_scan(args, table, mods):
    div = {k: g for k, g in table.items() if len(g) > 1}
    missing = {k: g for k, g in table.items()
               if len(set().union(*g.values())) < len(mods[k[0]])}
    print(f"{len(table)} shared functions across forks | "
          f"{len(div)} divergent | {len(missing)} not present in every fork\n")
    by_module = defaultdict(list)
    for (m, f), g in div.items():
        by_module[m].append((f, g))
    for m in sorted(by_module):
        print(f"--- {m}")
        for f, g in sorted(by_module[m], key=lambda x: -len(x[1])):
            print(f"  {f:<44} {len(g)} versions")
            for h, vs in sorted(g.items(), key=lambda kv: -len(kv[1])):
                tag = "majority" if len(vs) == max(len(x) for x in g.values()) else "        "
                print(f"      [{h}] {tag}  " + ", ".join(vs))
        print()


def cmd_accept(args, table, mods):
    out = Path(args.baseline)
    out.write_text(json.dumps(fingerprint(table), indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {out} — {len(table)} shared functions recorded. Commit this file.")


def cmd_check(args, table, mods):
    path = Path(args.baseline)
    if not path.exists():
        print(f"no {path}; run `accept` first.", file=sys.stderr)
        return 2
    old = json.loads(path.read_text(encoding="utf-8"))
    new = fingerprint(table)
    problems = []
    for key, groups in sorted(new.items()):
        if key not in old:
            if len(groups) > 1:
                problems.append(("NEW FUNCTION, already divergent", key, groups))
            continue
        if groups != old[key]:
            kind = ("DIVERGENCE INCREASED" if len(groups) > len(old[key])
                    else "DIVERGENCE RESOLVED" if len(groups) < len(old[key])
                    else "REGROUPED")
            problems.append((kind, key, groups))
    for key in sorted(set(old) - set(new)):
        problems.append(("FUNCTION GONE", key, old[key]))

    if not problems:
        print(f"ok — divergence matches {path} ({len(new)} shared functions)")
        return 0
    for kind, key, groups in problems:
        print(f"{kind}: {key}")
        for vs in groups:
            print("    " + ", ".join(vs))
    print(f"\n{len(problems)} change(s). Review, then `accept` if intended.")
    return 1


def cmd_show(args, table, mods):
    variants = mods.get(args.module, {})
    bodies = {v: fs[args.function] for v, fs in variants.items() if args.function in fs}
    if not bodies:
        print(f"{args.function} not found in any copy of {args.module}", file=sys.stderr)
        return 2
    groups = defaultdict(list)
    for v, src in bodies.items():
        groups[digest(src)].append(v)
    base = max(groups, key=lambda h: len(groups[h]))
    rep = {h: sorted(vs)[0] for h, vs in groups.items()}
    print(f"{args.module}::{args.function} — {len(groups)} versions\n")
    for h, vs in groups.items():
        print(f"  [{h}]{' majority' if h == base else '         '} " + ", ".join(vs))
    for h in groups:
        if h == base:
            continue
        print()
        print("\n".join(difflib.unified_diff(
            bodies[rep[base]].splitlines(), bodies[rep[h]].splitlines(),
            f"majority [{base}]", f"[{h}] {rep[h]}", lineterm="", n=2)))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="repository root (default: cwd)")
    ap.add_argument("--glob", action="append", default=None,
                    help="source pattern, repeatable (default: *.py)")
    ap.add_argument("--exclude", action="append", default=[],
                    help="glob of variant directories to ignore, repeatable — "
                         "e.g. --exclude '*/V0=*' to skip per-run copies")
    ap.add_argument("--baseline", default=BASELINE)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scan")
    sub.add_parser("accept")
    sub.add_parser("check")
    sp = sub.add_parser("show")
    sp.add_argument("module")
    sp.add_argument("function")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    mods = collect(root, args.glob or ["*.py"], tuple(args.exclude))
    table = survey(mods)
    return {"scan": cmd_scan, "accept": cmd_accept,
            "check": cmd_check, "show": cmd_show}[args.cmd](args, table, mods) or 0


if __name__ == "__main__":
    sys.exit(main())
