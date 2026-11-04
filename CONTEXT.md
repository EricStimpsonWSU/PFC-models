# CONTEXT

Standing brief for this repository. Read first; keep current. This file exists so
that neither you-after-a-break nor an assistant session has to re-derive context.

## What this repo is

The PFC model catalogue, installable as `pfc_models`. Shared substrate plus one
self-contained `simulation.py` per model variant.

## Conventions in force

- Substrate modules are **leaf modules with a documented contract, not base
  classes**. A spectral 2D periodic geometry and a finite-difference 2D
  non-periodic geometry are sibling files exposing the same attribute names —
  not a subclass pair.
- A new axis is added by adding a sibling file. Never by adding a flag to an
  existing substrate module to make it cover two cases.
- Model directory names state the physics plus any axis that departs from the
  defaults: 2D, spectral, fully periodic, float64, CuPy.
- Every model carries `model.toml` (declared axes), `model.md` (free energy,
  EOM, assumptions, provenance) and `test_regression.py` (fixed seed, asserted
  free energy at a fixed step).
- `variant_check.py` watches both `models/*/simulation.py` and sibling substrate
  modules for unpropagated changes.

## Decided

- Duplication across models is intentional and is not to be refactored into a
  hierarchy.
- Profiling instrumentation and prior-work reference implementations stay out of
  this repo (see README).
- The log-PFC nonlinear chemical potential clamps **the logarithm only**:
  `where(phi > minPhi, log(phi), minLog) + g*phi**2 + phi**3`. Clamping the full
  expression is a bug. Recorded in each log-family `model.md`.
- **2026-11-04.** Phase-diagram investigation is a cross-cutting concern homed
  here: linear-stability (spinodal) locus, lowest-energy ground states over
  candidate structures, strain-free lattice-constant (q0*) determination, the
  common-tangent / Maxwell construction and the standard diagram types. Which
  parameter regions, phases and coexistences to compute, and the resulting
  diagrams, belong to the paper or model study that uses them (log-PFC:
  `papers\2026-log-pfc`, `c2-phase-diagram`). Source: phase-diagram project,
  P02-01. Module layout not yet chosen; the code currently sits in the
  temporary `warrior\phase-diagram\`.
- **2026-11-04.** The phase-diagram minimiser (`MinimizeFreeEnergy`, now in
  `warrior\phase-diagram\_minimize_f.py`) ignores the `dt` passed to
  `_buildModel`, so rollback refinement never shrinks dt and termination happens
  through the exhausted-recursion exception. Documented in S01
  (`warrior\phase-diagram\reconciliation.md` §4); fixed when the minimiser
  moves here in S02 (pass dt through, explicit convergence criterion, test).

## Open

- Dimensionality: only 2D exists; 1D and 3D are planned and will add sibling
  geometry/kernel modules.
- Discretization: only spectral (fully periodic) exists; finite-difference with
  semi-periodic and non-periodic boundaries is planned. `discretization/fem/`
  is a placeholder for possible non-Eulerian work.
- Backends: `backends/torch_mps.py` is a stub for Mac GPU support.
