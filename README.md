# pfc-models

The catalogue. Two kinds of code, split on purpose:

- **Substrate** (`src/pfc_models/` minus `models/`) — geometry, kernels, backend
  dispatch, initial conditions, trackers, I/O, visualization. Single-sourced.
  Importing it is why the models no longer need a vendored copy of every helper.
- **Models** (`src/pfc_models/models/<name>/simulation.py`) — one
  self-contained implementation per variant. Duplication here is deliberate: a
  model is a readable statement of a free energy and its equations of motion,
  not a position in a class hierarchy.

The line between them: substrate is anything that does not change when the
physics changes. The moment a "shared" helper needs a flag to express a model's
physics, it belongs in the model.

## Catalogue

Generated — do not edit by hand:

```powershell
python tools\build_feature_matrix.py
```

writes `docs/feature-matrix.md` and `docs/feature-matrix.csv` from the
`model.toml` files, and prints the measured function-level divergence beside the
declared one.

## Why two matrices

The **declared** matrix (from `model.toml`) says how the models are supposed to
differ. The **measured** matrix (from `tools/variant_check.py`) says how their
code actually differs. A difference present in one and not the other is a
finding: either an undeclared feature, or a cross-cutting change that reached
some models and not others.

```powershell
python tools\variant_check.py --root src\pfc_models accept   # sign off today's divergence
python tools\variant_check.py --root src\pfc_models check    # exit 1 if it changed
```

Commit `.variant-baseline.json`; wire `check` into a pre-commit hook.

## Installing

```powershell
py -m venv .venv      ; .venv\Scripts\pip install -e .[gpu,viz,dev]      # GPU target
py -m venv .venvFFTW  ; .venvFFTW\Scripts\pip install -e .[cpu,viz,dev]  # CPU target
```

Papers depend on a **tag**, never on `main`:

```
pfc-models @ git+https://github.com/<user>/pfc-models@v0.4.2
```

## Not in this repo

- **Profiling instrumentation.** It is a testing concern and it makes model code
  harder to read. Profile in `sandbox/profiling/` against an installed tag.
- **Reference implementations of prior work** (logPFC, sHPFC). They live in
  `sandbox/references/` — read, compare, cite; do not vendor.
