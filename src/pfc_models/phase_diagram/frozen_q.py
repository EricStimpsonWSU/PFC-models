"""Frozen-field q0* minimisation: an alternative to `q0star.MinimizeFreeEnergy.Find_q0Star`.

For a field phi frozen on the grid, only the gradient term of f depends on the box
scale q (every wavevector scales as k = q k_ref), so f(q; phi) is known in closed form
and its minimiser q_opt(phi) is too (the adapter's `frozen_field`). At an equilibrium
relaxed at box scale q*, df_total/dq = df_frozen/dq (envelope theorem), so
q_opt(phi*) = q*. `MinimizeFrozenQ.solve` alternates: relax at q (Get_f_ABC: rollback,
adaptive dt, fitness/amplitude stops) to a loose fTol, jump q -> q_opt of the relaxed
field (the field is carried, the box rescaled), tighten fTol, and repeat until
|q_opt - q| < qTol at the final fTol with a converged stop. No q scan and no spline.

Differences from Find_q0Star (validated 2026-11-15, beta 0.25 and 0.5, logpfc_cpu):
where both accept a point they agree to |dq*| <= 3e-7, |df| <= 1.5e-11. Near branch
ends the state followed to q_opt can transform (fitness < requiredFitness) where
Find's fresh ansatz on a fixed q00 grid stays clean; such points are rejected here
(poor_fit) and the lower-f transformed state is not used. q* is not restricted to
Find's [0.8, 1.2] window; mask at analysis time if wanted.

`scan_density` runs one (structure, temperature) density list: independent solves
from q_start, then the edge pass: every non-ok point next to an ok one is retried
from a fresh ansatz started at the ok neighbour's q* (walking outward up to
`maxPast` consecutive failures), and an ok edge result replaces a non-ok original.

Status codes: 'ok', the stop names of q0star for a rejected relaxation
('poor_fit', 'min_amplitude', 'max_depth', 'nan'), 'not_converged' (q or f did not
settle within maxIter) and 'error' (exception).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np

from pfc_models.phase_diagram import q0star as _q0s

METHOD = 'frozen-q'
STATUS_OK = 'ok'
STATUS_NOT_CONVERGED = 'not_converged'
STATUS_ERROR = 'error'
DEFAULT_SCHEDULE = (1e-6, 1e-8, 1e-10, 1e-12)


class FrozenQRejected(Exception):
    """solve() did not give an accepted state; `status` holds the code."""

    def __init__(self, status, message=''):
        super().__init__(message or status)
        self.status = status


@dataclass(frozen=True)
class FrozenQSettings:
    """schedule: fTol per stage (the last one is the final tolerance; relax.fTol is
    not used); qTol: |q_opt - q| for convergence at the final stage; maxIter: relax/jump
    iterations; edge, maxPast: the edge pass of scan_density."""
    schedule: tuple = DEFAULT_SCHEDULE
    qTol: float = 1e-8
    maxIter: int = 40
    edge: bool = True
    maxPast: int = 3


class MinimizeFrozenQ:
    """Same constructor arguments as q0star.MinimizeFreeEnergy, with FrozenQSettings
    in place of QScanSettings. The adapter must provide `frozen_field(state, params,
    temp, cell)` (an object with f(q) and q_opt()); `defect_diagnostics(state, cell)`
    is used when present (record-only)."""

    def __init__(self, adapter, params, structure, cell, ansatz, relax, settings=None):
        if not hasattr(adapter, 'frozen_field'):
            raise ValueError(f'adapter {getattr(adapter, "SLUG", adapter)!r} has no frozen_field')
        self.adapter = adapter
        self.params = adapter.validate_params(params)
        self.structure = structure
        self.cell = adapter.validate_cell(cell)
        self.ansatz = adapter.validate_ansatz(ansatz)
        self.relax = relax
        self.settings = settings or FrozenQSettings()
        if len(self.settings.schedule) < 1:
            raise ValueError('schedule needs at least one fTol')
        self.last_result = None
        self.last_state = None

    def _inner(self, stage):
        s = self.settings.schedule
        fTol = float(s[stage])
        ATol = self.relax.ATol if stage == len(s) - 1 else max(self.relax.ATol, 1e6*fTol)
        rs = replace(self.relax, fTol=fTol, ATol=ATol)
        return _q0s.MinimizeFreeEnergy(self.adapter, self.params, self.structure, self.cell,
                                       self.ansatz, rs, _q0s.QScanSettings(1, 0.0, 1, 1.0))

    def solve(self, temp, meanDensity, q_start=1.0, state=None):
        """One density: returns a record dict (never raises on a rejected state;
        exceptions give status 'error'). Fields: status, q (q_opt of the final field,
        = q*), q_relaxed (box scale it was relaxed at), f (adapter f at q_relaxed),
        f_frozen (frozen f at q), fitness, A1..A3, stop, steps, iters, wall_s, q_start,
        message, plus defect diagnostics when the adapter has them."""
        t0 = time.perf_counter()
        st = self.settings
        q, carry, steps, stage = float(q_start), state, 0, 0
        rec = dict(method=METHOD, q_start=float(q_start), message='')
        last = None
        phi = None
        try:
            for it in range(st.maxIter):
                m = self._inner(stage)
                f, fit, *amps = m.Get_f_ABC(temp, meanDensity, q, state=carry)
                steps += m.last_steps
                phi = m.last_state
                ff = self.adapter.frozen_field(phi, self.params, temp, self.cell)
                qn = float(ff.q_opt())
                last = dict(q=q, q_opt=qn, f=float(f), f_frozen=float(ff.f(qn)), fitness=float(fit),
                            amps=[float(a) for a in amps], stop=m.last_stop, it=it)
                if m.last_stop not in (_q0s.STOP_CONVERGED, _q0s.STOP_TOTALT):
                    break
                dq = qn - q
                carry, q = phi, qn
                if stage < len(st.schedule) - 1:
                    stage += 1
                elif abs(dq) < st.qTol and m.last_stop == _q0s.STOP_CONVERGED:
                    break
        except Exception as e:  # noqa: BLE001
            rec.update(status=STATUS_ERROR, message=f'{type(e).__name__}: {e}', steps=steps,
                       wall_s=time.perf_counter() - t0)
            self.last_result, self.last_state = rec, None
            return rec
        ok = (last['stop'] == _q0s.STOP_CONVERGED and stage == len(st.schedule) - 1
              and abs(last['q_opt'] - last['q']) < st.qTol)
        if ok:
            status = STATUS_OK
        elif last['stop'] in (_q0s.STOP_CONVERGED, _q0s.STOP_TOTALT):
            status = STATUS_NOT_CONVERGED
        else:
            status = last['stop']
        rec.update(status=status, q=last['q_opt'], q_relaxed=last['q'], f=last['f'],
                   f_frozen=last['f_frozen'], fitness=last['fitness'], stop=last['stop'],
                   steps=steps, iters=last['it'] + 1)
        for j, a in enumerate(last['amps']):
            rec[f'A{j + 1}'] = a
        dd = getattr(self.adapter, 'defect_diagnostics', None)
        if dd is not None and phi is not None:
            rec.update(dd(phi, self.cell))
        rec['wall_s'] = time.perf_counter() - t0
        self.last_result, self.last_state = rec, phi
        return rec

    def Find_q0Star(self, temp, meanDensity, q_start=1.0, record=None):
        """(q0*, fmin) like MinimizeFreeEnergy.Find_q0Star; raises FrozenQRejected
        when the state is not accepted. The full record is in `last_result` and is
        passed to `record` when given."""
        r = self.solve(temp, meanDensity, q_start)
        if record is not None:
            record(r)
        if r['status'] != STATUS_OK:
            raise FrozenQRejected(r['status'], r.get('message', ''))
        return r['q'], r['f']


# ---------------------------------------------------------------- edge pass

def _is_ok(r):
    return r is not None and r.get('status') == STATUS_OK


def edge_walks(grid, results):
    """(i, d) for every ok grid index i whose neighbour i+d (d = -1, +1) is not ok.
    `grid`: sorted densities; `results`: density -> record."""
    out = []
    for i, p in enumerate(grid):
        if not _is_ok(results.get(p)):
            continue
        for d in (-1, 1):
            j = i + d
            if 0 <= j < len(grid) and not _is_ok(results.get(grid[j])):
                out.append((i, d))
    return out


def edge_walk(solve, grid, results, i, d, maxPast=3):
    """Walk outward from ok index i in direction d. Each non-ok point is rerun by
    solve(phi, q_start) from a fresh ansatz with q_start = q* of grid[i]; the walk
    stops at an originally ok point, the grid end, or after maxPast consecutive
    non-ok results. Returns records with q_start, edge_from, edge_dir added."""
    q_seed = float(results[grid[i]]['q'])
    out, fails, j = [], 0, i + d
    while 0 <= j < len(grid) and fails < maxPast:
        p = grid[j]
        if _is_ok(results.get(p)):
            break
        r = dict(solve(p, q_seed))
        r.update(phi=p, q_start=q_seed, edge_from=grid[i], edge_dir=d, pass_='edge')
        out.append(r)
        fails = 0 if _is_ok(r) else fails + 1
        j += d
    return out


def merge_edge(results, edge_records):
    """Copy of `results` where an ok edge record replaces a non-ok original (lowest f
    among several ok edge records for one density)."""
    out = dict(results)
    for r in edge_records:
        if not _is_ok(r):
            continue
        p = r['phi']
        cur = out.get(p)
        if not _is_ok(cur) or (cur.get('pass_') == 'edge' and r['f'] < cur['f']):
            out[p] = r
    return out


def scan_density(minimizer, temp, densities, q_start=1.0, record=None, done=None):
    """One structure at one temperature over `densities` (sorted ascending).

    Pass 1: minimizer.solve(temp, phi, q_start) per density (skipping those in `done`,
    a dict density -> pass-1 record from a checkpoint). Pass 2 (settings.edge): edge
    walks. `record(rec)` is called after every solve (pass 1 and edge), for
    checkpointing. Returns (final, pass1, edge): final maps density -> the merged
    record (pass_ 'indep' or 'edge'), pass1 the independent records, edge the list of
    edge-walk records."""
    grid = [float(p) for p in densities]
    if any(b <= a for a, b in zip(grid, grid[1:])):
        raise ValueError('densities must be strictly increasing')
    pass1 = dict(done or {})
    for p in grid:
        if p in pass1:
            continue
        r = minimizer.solve(temp, p, q_start)
        r.update(phi=p, pass_='indep')
        pass1[p] = r
        if record is not None:
            record(r)
    edge = []
    if minimizer.settings.edge:
        def solve(p, q):
            r = minimizer.solve(temp, p, q)
            if record is not None:
                record({**r, 'phi': p, 'pass_': 'edge'})
            return r
        for i, d in edge_walks(grid, pass1):
            edge += edge_walk(solve, grid, pass1, i, d, minimizer.settings.maxPast)
    return merge_edge(pass1, edge), pass1, edge
