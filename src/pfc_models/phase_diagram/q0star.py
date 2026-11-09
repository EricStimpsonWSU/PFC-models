"""Strain-free lattice scale q0* and fmin(phibar) for one structure, model-agnostic.

Contract
--------
`MinimizeFreeEnergy` drives a model through the adapter module
`pfc_models.models.<slug>.phase_diagram` and calls only these names on it:

    validate_params, validate_cell, validate_ansatz, STRUCTURES, SLUG
    build(params, structure, cell, ansatz, temp, meanDensity, q00, dt,
          state=None, nThreads=1) -> sim
    get_state(sim) / set_state(sim, state)      opaque copy of the conserved fields
    get_time(sim) / set_time(sim, t)
    get_dt(sim), step(sim), free_energy(sim)
    diagnose(sim, structure, cell) -> Diagnostics

`temp` (the temperature-like parameter) and `meanDensity` (whatever the model's
conserved densities are) are passed through untouched; so are the state returned
by `get_state`, the structure and the cell. Nothing here knows a k-vector, the
number of fields or the cell geometry. Of the diagnostics only `fitness` and
`amplitudes[0]` (the primary amplitude) are read.

Known limitation: the scan variable is ONE isotropic lattice scale `q00`, which
the adapter turns into a box. Cells that need several relaxed cell parameters
(anisotropic lattices, non-cubic 3D cells) need a sibling search module, not a
flag on this one.

Provenance
----------
`MinimizeFreeEnergy` is relocated from warrior/phase-diagram/_minimize_f.py
(S02, P02-02; design note sec. 3). `_generateSim`, `_takeStep`, `Get_f_ABC` and
`Find_q0Star` keep their names and bodies; calls on `Simulation.py` objects became
adapter calls. The model-specific methods (`_buildModel`, `_buildSystemGeometry`,
`_build3Mode`, `_calcFitness`) moved to the adapters. `Get_f` (no caller outside
a notebook that is not carried) and `temperature_domain_idx` (never read) are
dropped. Two mechanical changes were needed because the generic code cannot
read a `Simulation` attribute: `_takeStep` takes `temp` (the original read
the epsilon stored on the Simulation, which holds the same value) and the rollback copy of the
state is taken with `get_state` instead of `ndarray.copy`.

Known defects, carried unchanged in this relocation (commit A) so that results
stay bit-identical to the original; each is one local hunk for commit B:

* D1  `_generateSim` builds with `self.dt` and ignores the `dt` it is given, so
      the rollback refinement in `_takeStep` never shrinks the time step
      (P02-01 reconciliation sec. 4).
* D2  `Get_f_ABC` has no convergence criterion; it stops when one stepT block
      fails to lower f at round-off and the rollback recursion exhausts
      (`Max recurrsion depth reached`), an exception the broad `except` turns
      into `break` (P02-01 sec. 4). The same `except` hides genuine errors.
* D3  `_takeStep` subdivides by its `divs=10` default; `RelaxSettings.divs` is
      never passed down.
* D4  With `requiredFitness=None`, `Get_f_ABC` reads an unbound `A`; the
      NameError is swallowed by the broad `except` after one block.
* D5  `Find_q0Star` counts points that pass fitness AND amplitude, but fits the
      spline and takes the argmin over fitness only.
* D6  `Find_q0Star` takes the argmin over the filtered f values and uses that
      index on the unfiltered q00 range.
* D7  `Find_q0Star` accepts any root of the spline derivative (maxima and
      inflections included); the nearest root to the mis-indexed q00_min wins.
* D8  "Insufficient points" fires at `size <= minPoints`, i.e. minPoints
      accepted points are not enough.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

# numerics
import numpy as np

# Curve fitting
from scipy.interpolate import CubicSpline


# Find_q0Star messages; scan_density maps them to status codes
MSG_INSUFFICIENT_POINTS = 'Insufficient points found that satisfy fitness constraints.'
MSG_NO_LOCAL_MINIMUM = 'No local minimum found for q00.'

STATUS_OK = 'ok'
STATUS_INSUFFICIENT_POINTS = 'insufficient_points'
STATUS_NO_LOCAL_MINIMUM = 'no_local_minimum'
STATUS_ERROR = 'error'

FMIN_FILE = 'fmin.npz'
QSCAN_FILE = 'qscan.npz'


@dataclass(frozen=True)
class RelaxSettings:
    """Relaxation by time stepping (legacy `run_vals` entry; `minimum-A` -> `minimumA`)."""
    dt: float
    totalT: float
    stepT: float
    divs: int
    maxDepth: int
    requiredFitness: float | None
    minimumA: float


@dataclass(frozen=True)
class QScanSettings:
    """q00 scan with spline refinement (legacy `fit_vals` entry, renamed to the
    `MinimizeFreeEnergy` attributes: `init-delta-q` -> `Deltaq`, `q-steps` ->
    `q00_Steps`, `pass-deltaq-refinement` -> `Deltaq_Refinement`)."""
    passes: int
    Deltaq: float
    q00_Steps: int
    Deltaq_Refinement: float
    q00_start: float = 1.0
    minPoints: int = 4


@dataclass(frozen=True)
class Diagnostics:
    """What an adapter's `diagnose` returns. The generic code reads `fitness`
    and `amplitudes[0]`; `details` is free-form and adapter-defined."""
    fitness: float
    amplitudes: tuple[float, ...]
    details: dict = field(default_factory=dict)


# Minimize free energy using simulation
class MinimizeFreeEnergy():
    def __init__(
        self,
        adapter,
        params,
        structure,
        cell,
        ansatz,
        relax,
        qscan,
        verbose=False,
    ):
        self.adapter = adapter

        # Model, structure, cell and ansatz: schema owned by the adapter
        self.params = adapter.validate_params(params)
        self.structure = adapter.STRUCTURES[structure] if isinstance(structure, str) else structure
        self.cell = adapter.validate_cell(cell)
        self.ansatz = adapter.validate_ansatz(ansatz)

        # Run parameters
        self.relax = relax
        self.dt = relax.dt
        self.totalT = relax.totalT
        self.stepT = relax.stepT
        self.divs = relax.divs
        self.maxDepth = relax.maxDepth
        self.requiredFitness = relax.requiredFitness
        self.minimumA = relax.minimumA

        # q0* fitting parameters
        self.qscan = qscan
        self.passes = qscan.passes
        self.Deltaq = qscan.Deltaq
        self.q00_Steps = qscan.q00_Steps
        self.Deltaq_Refinement = qscan.Deltaq_Refinement
        self.q00_start = qscan.q00_start
        self.minPoints = qscan.minPoints

        # Using...
        if verbose:
            print(f'Using model: {adapter.SLUG}, parameters: {self.params}')
            print(f'Using structure: {getattr(self.structure, "name", self.structure)}, cell: {self.cell}')
            print(f'Using ansatz: {self.ansatz}')
            print(f'Using run parameters: {relax}')
            print(f'Using fit-vals: {qscan}')

    # Generate simulation:
    def _generateSim(self, temp, meanDensity, q00, dt, densityField=None, verbose=True):
        # build model, geometry and initial state (adapter).
        # D1: built with self.dt; the dt argument is ignored, as in the original _buildModel.
        sim = self.adapter.build(
            self.params, self.structure, self.cell, self.ansatz,
            temp, meanDensity, q00, self.dt,
            state=densityField,
            nThreads=1,
        )

        # Built...
        if verbose:
            print(f'Built {self.adapter.SLUG}: temp={temp}, meanDensity={meanDensity}, q00={q00}, dt={dt:0.4e} (sim dt={self.adapter.get_dt(sim):0.4e})')

        return sim

    # Take step (recursively up to max depth, default 3)
    def _takeStep(self, sim, temp, meanDensity, q00, stepT, divs=10, depth=0, maxDepth=None, verbose=True):
        if maxDepth is None:
            maxDepth = self.maxDepth
        ad = self.adapter

        dt = ad.get_dt(sim)
        miniSteps = int(stepT/dt)
        phiLast = ad.get_state(sim)
        tLast = ad.get_time(sim)
        fLast = ad.free_energy(sim)
        for _ in range(miniSteps):
            ad.step(sim)

        f = ad.free_energy(sim)
        delta_f = f - fLast
        t = ad.get_time(sim)

        if delta_f >= 0 or np.isnan(f):
            # revert density field and time
            if verbose:
                print(f'rollback: t={t:0.6f}, Δt={ad.get_dt(sim):0.3e}, f={f:0.18f}, Δf={delta_f:0.5e}')
            ad.set_state(sim, phiLast)
            ad.set_time(sim, tLast)

            if depth >= maxDepth:
                raise Exception(f'Max recurrsion depth reached at t={ad.get_time(sim):0.6f}')

            _stepT = stepT / divs
            _dt = dt / divs
            _phiLast = ad.get_state(sim)
            _sim = self._generateSim(temp, meanDensity, q00, _dt, _phiLast, verbose=verbose)
            ad.set_time(_sim, ad.get_time(sim))
            _fLast = fLast

            if verbose:
                print(f'here: depth={depth}, dt={dt:0.3e}, stepT={_stepT:0.3e}')
            for _ in range(divs):
                try:
                    self._takeStep(sim=_sim, temp=temp, meanDensity=meanDensity, q00=q00, stepT=_stepT, divs=divs, depth=depth+1, maxDepth=maxDepth, verbose=verbose)
                except Exception as e:
                    raise(e)

                ad.set_state(sim, ad.get_state(_sim))
                ad.set_time(sim, ad.get_time(_sim))
            return


        if verbose:
            print(f't={t:0.6f}, Δt={ad.get_dt(sim):0.3e}, f={f:0.18f}, Δf={delta_f:0.5e}')
        return

    # Get the f(meanDensity, q00) using system parameters specified in constructor
    def Get_f_ABC(self, temp, meanDensity, q00, maxDepth=None, verbose=False):
        """Relax the ansatz at fixed box scale q00; return (f, fitness, *amplitudes)."""
        if maxDepth is None:
            maxDepth = self.maxDepth
        ad = self.adapter

        totalT = self.totalT
        stepT = self.stepT
        dt = self.dt
        steps = int(totalT/stepT)
        miniSteps = int(np.ceil(stepT/dt))
        _dt = stepT/miniSteps # keep stepT exact

        # initialize simulation
        sim = self._generateSim(temp, meanDensity, q00, _dt, densityField=None, verbose=verbose)
        fLast = ad.free_energy(sim)

        for _ in range(steps):
            try:
                self._takeStep(sim=sim, temp=temp, meanDensity=meanDensity, q00=q00, stepT=stepT, maxDepth=maxDepth, verbose=verbose)
                fLast = ad.free_energy(sim)
                if verbose:
                    print(fLast)

                if not self.requiredFitness is None:
                    diag = ad.diagnose(sim, self.structure, self.cell)
                    fitness, A = diag.fitness, diag.amplitudes[0]
                    if fitness < self.requiredFitness:
                        if verbose:
                            print(f'Poor fit: fitness={fitness:0.2f}')
                        break
                if A < self.minimumA:
                    if verbose:
                        print(f'Poor fit: A={A:0.2e} < min A')
                    break
            except Exception as e:
                if verbose:
                    print(f'Exception: {e}')
                break

        f = ad.free_energy(sim)
        diag = ad.diagnose(sim, self.structure, self.cell)
        return (f, diag.fitness, *diag.amplitudes)

    # Find q0* (q00 that minimizes f)
    def Find_q0Star(self, temp, meanDensity, minPoints=None, verbose=False, record=None):
        """q00 that minimises f at (temp, meanDensity); returns (q0*, fmin).

        Raises Exception(MSG_INSUFFICIENT_POINTS) or Exception(MSG_NO_LOCAL_MINIMUM)
        as the original did. `minPoints=None` takes `QScanSettings.minPoints`.
        `record`, if given, is called once per Get_f_ABC evaluation with a dict
        (pass, q00, f, fitness, amplitudes); it does not affect the result.
        """
        if minPoints is None:
            minPoints = self.minPoints
        passes = self.passes
        Deltaq = self.Deltaq
        q00_Steps = self.q00_Steps
        Deltaq_Refinement = self.Deltaq_Refinement

        q00_best = self.q00_start
        for pass_idx in range(passes):
            q00_range = np.linspace(q00_best - q00_Steps//2 * Deltaq, q00_best + q00_Steps//2 * Deltaq, q00_Steps, endpoint = True)
            f_values = np.zeros_like(q00_range)
            fit_values = np.zeros_like(q00_range)
            A_values = np.zeros_like(q00_range)
            for i, q00 in enumerate(q00_range):
                out = self.Get_f_ABC(temp=temp, meanDensity=meanDensity, q00=q00, verbose=verbose)
                f_values[i], fit_values[i], A_values[i] = out[0], out[1], out[2]
                if record is not None:
                    record({'pass': pass_idx, 'q00': float(q00), 'f': float(out[0]),
                            'fitness': float(out[1]), 'amplitudes': tuple(float(a) for a in out[2:])})

            # D5: the count uses fitness AND amplitude, the fit below uses fitness only
            counted = (fit_values>=self.requiredFitness) & (A_values>=self.minimumA)
            fitted = fit_values>=self.requiredFitness

            # D8: minPoints accepted points are reported as insufficient
            if q00_range[counted].size <= minPoints:
                raise Exception(MSG_INSUFFICIENT_POINTS)

            # D6: index into the filtered values, applied to the unfiltered range
            q00_min_idx = np.argmin(f_values[fitted])
            q00_min = q00_range[q00_min_idx]

            f_q00_spl = CubicSpline(q00_range[fitted], f_values[fitted])
            # D7: every root of f', maxima and inflections included
            roots = f_q00_spl.derivative().roots(extrapolate=False)

            if roots.size == 0:
                raise Exception(MSG_NO_LOCAL_MINIMUM)

            q00_best = np.abs(min(roots, key=lambda x: abs(x - q00_min)))
            Deltaq *= Deltaq_Refinement
        return q00_best, f_q00_spl(q00_best)


def status_from_exception(exc):
    """Map an exception raised by Find_q0Star to (status code, message)."""
    text = str(exc)
    if text == MSG_INSUFFICIENT_POINTS:
        return STATUS_INSUFFICIENT_POINTS, text
    if text == MSG_NO_LOCAL_MINIMUM:
        return STATUS_NO_LOCAL_MINIMUM, text
    return STATUS_ERROR, f'{type(exc).__name__}: {text}'


def _save_npz_atomic(path, arrays):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


def _fmin_arrays(phi, q0star, fmin, status, message, wall_s):
    return {
        'phi': np.asarray(phi, dtype=float),
        'q0star': np.asarray(q0star, dtype=float),
        'fmin': np.asarray(fmin, dtype=float),
        'status': np.asarray(status, dtype=str),
        'message': np.asarray(message, dtype=str),
        'wall_s': np.asarray(wall_s, dtype=float),
    }


def _qscan_arrays(rows, n_amplitudes):
    k = n_amplitudes
    if rows:
        k = max(len(r['amplitudes']) for r in rows)
    amplitudes = np.full((len(rows), k), np.nan)
    for j, r in enumerate(rows):
        amplitudes[j, :len(r['amplitudes'])] = r['amplitudes']
    return {
        'phi_index': np.asarray([r['phi_index'] for r in rows], dtype=np.int64),
        'pass_index': np.asarray([r['pass_index'] for r in rows], dtype=np.int64),
        'q00': np.asarray([r['q00'] for r in rows], dtype=float),
        'f': np.asarray([r['f'] for r in rows], dtype=float),
        'fitness': np.asarray([r['fitness'] for r in rows], dtype=float),
        'amplitudes': amplitudes,
    }


def _load_checkpoint(checkpoint, phi):
    """Previous per-density results and q-scan rows, or None if there are none."""
    fmin_path = os.path.join(checkpoint, FMIN_FILE)
    if not os.path.exists(fmin_path):
        return None
    with np.load(fmin_path, allow_pickle=False) as z:
        old = {k: z[k] for k in z.files}
    if old['phi'].shape != phi.shape or not np.array_equal(old['phi'], phi):
        raise ValueError(f'{fmin_path} holds a different density list; refusing to resume into it')
    done = {i for i, s in enumerate(old['status']) if str(s)}
    rows = []
    qscan_path = os.path.join(checkpoint, QSCAN_FILE)
    if os.path.exists(qscan_path):
        with np.load(qscan_path, allow_pickle=False) as z:
            q = {k: z[k] for k in z.files}
        for j in range(q['q00'].size):
            i = int(q['phi_index'][j])
            if i not in done:  # rows of a density that never got a status are redone
                continue
            amps = q['amplitudes'][j]
            rows.append({'phi_index': i, 'pass_index': int(q['pass_index'][j]), 'q00': float(q['q00'][j]),
                         'f': float(q['f'][j]), 'fitness': float(q['fitness'][j]),
                         'amplitudes': tuple(float(a) for a in amps[~np.isnan(amps)])})
    return old, done, rows


def scan_density(minimizer, temp, densities, checkpoint=None, verbose=False):
    """fmin and q0* over a density list: one task (one structure, one temperature).

    Densities are independent: a failure costs that density only. Per density
    the result holds phi, q0star, fmin, status, message, wall_s (q0star and fmin
    NaN unless status is 'ok'), and every Get_f_ABC evaluation is kept as a
    q-scan row (phi_index, pass_index, q00, f, fitness, amplitudes).

    Status codes: 'ok'; 'insufficient_points' and 'no_local_minimum' (the two
    Find_q0Star exceptions, message verbatim); 'error' (anything else, message
    'Type: text').

    With `checkpoint` (an existing directory) fmin.npz and qscan.npz are
    rewritten atomically (os.replace) after every density, and a restart skips
    the densities that already have a status. Returns a dict of the fmin.npz
    arrays plus 'qscan', a dict of the qscan.npz arrays.
    """
    phi = np.asarray(list(densities), dtype=float)
    n = phi.size
    q0star = [np.nan] * n
    fmin = [np.nan] * n
    status = [''] * n
    message = [''] * n
    wall_s = [np.nan] * n
    rows = []
    done = set()

    if checkpoint is not None:
        prev = _load_checkpoint(checkpoint, phi)
        if prev is not None:
            old, done, rows = prev
            for i in done:
                q0star[i] = float(old['q0star'][i])
                fmin[i] = float(old['fmin'][i])
                status[i] = str(old['status'][i])
                message[i] = str(old['message'][i])
                wall_s[i] = float(old['wall_s'][i])

    n_amplitudes = 0
    for i in range(n):
        if i in done:
            continue
        recs = []
        t0 = time.perf_counter()
        try:
            q, f = minimizer.Find_q0Star(temp=temp, meanDensity=phi[i], verbose=verbose, record=recs.append)
            q0star[i], fmin[i] = float(q), float(f)
            status[i], message[i] = STATUS_OK, ''
        except Exception as e:
            status[i], message[i] = status_from_exception(e)
        wall_s[i] = time.perf_counter() - t0
        for r in recs:
            rows.append({'phi_index': i, 'pass_index': r['pass'], 'q00': r['q00'], 'f': r['f'],
                         'fitness': r['fitness'], 'amplitudes': r['amplitudes']})
            n_amplitudes = max(n_amplitudes, len(r['amplitudes']))
        if verbose:
            print(f'phi={phi[i]}: {status[i]} q0*={q0star[i]} fmin={fmin[i]} ({wall_s[i]:0.1f} s) {message[i]}')

        if checkpoint is not None:
            # q-scan first: a crash between the two writes leaves rows without a
            # status, which a resume drops and redoes
            _save_npz_atomic(os.path.join(checkpoint, QSCAN_FILE), _qscan_arrays(rows, n_amplitudes))
            _save_npz_atomic(os.path.join(checkpoint, FMIN_FILE),
                             _fmin_arrays(phi, q0star, fmin, status, message, wall_s))

    result = _fmin_arrays(phi, q0star, fmin, status, message, wall_s)
    result['qscan'] = _qscan_arrays(rows, n_amplitudes)
    return result
