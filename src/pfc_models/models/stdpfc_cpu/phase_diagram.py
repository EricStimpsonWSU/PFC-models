"""Phase-diagram adapter for stdpfc_cpu: the model side of pfc_models.phase_diagram.

Sibling of `models/logpfc_cpu/phase_diagram.py`, same names (design note sec. 2):

    SLUG, STRUCTURES, validate_params, validate_cell, validate_ansatz
    build, get_state, set_state, get_time, set_time, get_dt, step, free_energy
    diagnose
    liquid_free_energy, liquid_chemical_potential
    spinodal, critical_point, liquid_spinodal, existence_bounds, scan_window

Model (model.md): f = n beta/2 (q0^2 + lap)^2 n + eps/2 n^2 + g/3 n^3 + v0/4 n^4
with v0 = 1 hard-wired in `simulation.py`. The field is `n`, a density deviation
that may take either sign; `temp` is epsilon and `meanDensity` is nbar (a float).
Every analytic name has a closed form here.

Differences from the log-PFC adapter, all of them consequences of n having no
positivity bound:

* no rescale of the ansatz;
* the ansatz amplitude is amplitudeSign * amplitudeScale * A_i, an absolute
  amplitude that does not involve meanDensity. Triangle and honeycomb share one
  scale, so at g = 0 (where f[-n] = f[n]) the triangle at -nbar is the exact
  mirror of the honeycomb at +nbar;
* the fitness amplitudes are absolute (not divided by meanDensity), so
  `RelaxSettings.minimumA` is an absolute amplitude for this model;
* `scan_window` does not clamp the window to positive densities.

Structures, k-vector conventions and the rectangular-cell box are the log
adapter's (legacy R2C `lattice_vals`; the C2C sets are not carried).
"""
from __future__ import annotations

from dataclasses import dataclass

# numerics
import numpy as np
import pyfftw

from pfc_models.models.stdpfc_cpu import simulation as _simulation
from pfc_models.phase_diagram.q0star import Diagnostics

SLUG = 'stdpfc_cpu'


@dataclass(frozen=True)
class Structure:
    """One target lattice. `kvectors` = (1-mode, 2-mode, 3-mode) sets of R2C
    (row, col) indices for one rectangular cell; `amplitudeScale` is an absolute
    amplitude; `maxima_per_cell` is the expected count of density maxima per
    rectangular cell (None: ridges, not points)."""
    name: str
    kvectors: tuple
    amplitudeSign: int
    amplitudeScale: float
    maxima_per_cell: int | None


STRUCTURES = {
    'triangle': Structure(
        name='triangle',
        kvectors=(
            ((1, 1), (2, 0), (-2, 0), (-1, 1)),
            ((0, 2), (3, 1), (-3, 1)),
            ((2, 2), (4, 0), (-4, 0), (-2, 2)),
        ),
        amplitudeSign=+1,
        amplitudeScale=1.0/3,
        maxima_per_cell=2,
    ),
    'honeycomb': Structure(
        name='honeycomb',
        kvectors=(
            ((1, 1), (2, 0), (-2, 0), (-1, 1)),
            ((0, 2), (3, 1), (-3, 1)),
            ((2, 2), (4, 0), (-4, 0), (-2, 2)),
        ),
        amplitudeSign=-1,
        amplitudeScale=1.0/3,
        maxima_per_cell=4,
    ),
    'stripe': Structure(
        name='stripe',
        kvectors=(
            ((2, 0), (-2, 0)),
            ((4, 0), (-4, 0)),
            ((6, 0), (-6, 0)),
        ),
        amplitudeSign=+1,
        amplitudeScale=1.0/2,
        maxima_per_cell=None,
    ),
}

_PARAM_KEYS = ('beta', 'g', 'v0', 'Gamma', 'q0')
_CELL_KEYS = ('uNx', 'uNy', 'Mx', 'My')
_ANSATZ_KEYS = ('mode', 'A1', 'A2', 'A3')


# ---------------------------------------------------------------- validation

def validate_params(params):
    """beta, g required; v0 must be 1.0 (hard-wired in simulation.py); Gamma, q0 default 1.0."""
    p = dict(params)
    unknown = sorted(set(p) - set(_PARAM_KEYS))
    if unknown:
        raise ValueError(f'{SLUG}: unknown model parameters {unknown}; allowed {list(_PARAM_KEYS)}')
    missing = [k for k in ('beta', 'g') if k not in p]
    if missing:
        raise ValueError(f'{SLUG}: missing model parameters {missing}')
    v0 = float(p.get('v0', 1.0))
    if v0 != 1.0:
        raise ValueError(f'{SLUG}: v0={v0} not supported; the quartic coefficient is hard-wired to 1 in simulation.py')
    return {
        'beta': float(p['beta']),
        'g': float(p['g']),
        'v0': 1.0,
        'Gamma': float(p.get('Gamma', 1.0)),
        'q0': float(p.get('q0', 1.0)),
    }


def validate_cell(cell):
    """uNx, uNy: grid points per rectangular unit cell; Mx, My: cells per box side."""
    c = dict(cell)
    if sorted(c) != sorted(_CELL_KEYS):
        raise ValueError(f'{SLUG}: cell needs exactly {list(_CELL_KEYS)}, got {sorted(c)}')
    out = {}
    for k in _CELL_KEYS:
        v = c[k]
        if int(v) != v or int(v) < 1:
            raise ValueError(f'{SLUG}: cell {k}={v!r} must be a positive integer')
        out[k] = int(v)
    return out


def validate_ansatz(ansatz):
    """mode (label, e.g. 'L2H', 'H2L'); A1, A2, A3: 1-, 2-, 3-mode amplitudes in units of amplitudeScale."""
    a = dict(ansatz)
    if sorted(a) != sorted(_ANSATZ_KEYS):
        raise ValueError(f'{SLUG}: ansatz needs exactly {list(_ANSATZ_KEYS)}, got {sorted(a)}')
    if not isinstance(a['mode'], str):
        raise ValueError(f'{SLUG}: ansatz mode must be a string label')
    return {'mode': a['mode'], 'A1': float(a['A1']), 'A2': float(a['A2']), 'A3': float(a['A3'])}


def _structure(structure):
    return STRUCTURES[structure] if isinstance(structure, str) else structure


# ---------------------------------------------------------------- cell, ansatz, fitness

def _mode_indices(structure, cell):
    # ...1,2,3-mode k-vector indices, scaled to the Mx x My supercell
    out = []
    for pos in structure.kvectors:
        rows, cols = np.array(pos).T
        out.append((rows * cell['My'], cols * cell['Mx']))
    return out


def _box(cell, q00):
    # rectangular cell of the triangular lattice with first shell at q00
    Nx, Ny = cell['uNx']*cell['Mx'], cell['uNy']*cell['My']
    a = 4*np.pi / (3**0.5 * q00)
    h = 4*np.pi / q00
    Lx, Ly = cell['Mx']*a, cell['My']*h
    return _simulation.Build2DGeometry(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly)


# Build 3-mode state for initial conditions and fitness evaluation (no positivity rescale)
def _build3Mode(geometry, structure, cell, meanDensity, A, B, C):
    X, Y = geometry.X, geometry.Y
    Nx, Ny = cell['uNx']*cell['Mx'], cell['uNy']*cell['My']
    (rows_1mode, cols_1mode), (rows_2mode, cols_2mode), (rows_3mode, cols_3mode) = _mode_indices(structure, cell)

    n_hat = np.zeros_like(X, dtype=complex)
    n_hat[0,0] = meanDensity*X.size
    n_hat[rows_1mode, cols_1mode] = A*X.size
    n_hat[rows_2mode, cols_2mode] = B*X.size
    n_hat[rows_3mode, cols_3mode] = C*X.size

    return np.fft.irfftn(n_hat, s=(Ny,Nx), axes=(0,1)).real


def _refresh_n_hat(sim):
    # the FFT CalculateFreeEnergy performs, so diagnose does not depend on call order
    n_copy = pyfftw.byte_align(np.copy(sim.State.n))
    sim.fft_fwd(n_copy, sim.State.n_hat)
    return sim.State.n_hat


# Fitness: 1 - mean squared deviation of n from its 3-mode reconstruction; absolute amplitudes
def _calcFitness(sim, structure, cell):
    n = sim.n
    n_hat = _refresh_n_hat(sim)
    Nx, Ny = cell['uNx']*cell['Mx'], cell['uNy']*cell['My']
    simSize = Nx * Ny
    meanDensity = n_hat.flat[0]/simSize
    (rows_1mode, cols_1mode), (rows_2mode, cols_2mode), (rows_3mode, cols_3mode) = _mode_indices(structure, cell)

    A = structure.amplitudeSign * np.nanmean(np.abs(n_hat[rows_1mode, cols_1mode])) / simSize
    B = structure.amplitudeSign * np.nanmean(np.abs(n_hat[rows_2mode, cols_2mode])) / simSize
    C = structure.amplitudeSign * np.nanmean(np.abs(n_hat[rows_3mode, cols_3mode])) / simSize
    n_3mode = _build3Mode(sim.Geometry, structure, cell, meanDensity, A, B, C)
    return 1 - float(np.abs(np.power(n - n_3mode, 2).mean())), np.abs(A), np.abs(B), np.abs(C)


def _maxima_per_cell(field, cell):
    # strict local maxima on the periodic grid (8 neighbours), per rectangular cell
    is_max = np.ones(field.shape, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                is_max &= field > np.roll(field, (dy, dx), axis=(0, 1))
    return int(is_max.sum()) / (cell['Mx']*cell['My'])


# ---------------------------------------------------------------- simulation contract

def build(params, structure, cell, ansatz, temp, meanDensity, q00, dt, state=None, nThreads=1):
    """Model + box at q00 + (ansatz if state is None, else a copy of state) -> Simulation."""
    params = validate_params(params)
    structure = _structure(structure)
    cell = validate_cell(cell)
    ansatz = validate_ansatz(ansatz)

    model = _simulation.BuildStdPFCModelParameters(
        Gamma = params['Gamma'],
        beta = params['beta'],
        q0 = params['q0'],
        epsilon = temp,
        g = params['g'],
        dt = dt,
        meanDensity = meanDensity
    )
    geometry = _box(cell, q00)

    if state is None:
        n0 = _build3Mode(
            geometry, structure, cell,
            meanDensity,
            structure.amplitudeSign*structure.amplitudeScale*ansatz['A1'],
            structure.amplitudeSign*structure.amplitudeScale*ansatz['A2'],
            structure.amplitudeSign*structure.amplitudeScale*ansatz['A3']
        )
    else:
        n0 = state

    sim = _simulation.BuildNumpyCPU_Simulation(
        modelParameters = model,
        geometry = geometry,
        nThreads = nThreads
    )
    sim.State.n[...] = n0
    return sim


def get_state(sim):
    return sim.n.copy()


def set_state(sim, state):
    sim.n[...] = state


def get_time(sim):
    return sim.t


def set_time(sim, t):
    sim.State.t = t


def get_dt(sim):
    return sim.Model.dt


def step(sim):
    sim.TimeStep()


def free_energy(sim):
    return sim.CalculateFreeEnergy()


def diagnose(sim, structure, cell):
    """fitness = 1 - mean((n - n_3mode)^2); amplitudes (A, B, C) = mean |n_hat| over
    the 1-, 2-, 3-mode sets (absolute); details: maxima_per_cell."""
    structure = _structure(structure)
    cell = validate_cell(cell)
    fitness, A, B, C = _calcFitness(sim, structure, cell)
    return Diagnostics(
        fitness=fitness,
        amplitudes=(A, B, C),
        details={'maxima_per_cell': _maxima_per_cell(sim.n, cell)},
    )


# ---------------------------------------------------------------- uniform state, analytic

def liquid_free_energy(n, params, temp):
    """Uniform f: (beta q0^4 + eps)/2 n^2 + g/3 n^3 + v0/4 n^4."""
    p = validate_params(params)
    beta, g, v0, q0 = p['beta'], p['g'], p['v0'], p['q0']
    return (
        1./2*(beta*q0**4 + temp)*n**2 +
        1./3*g*n**3 +
        1./4*v0*n**4
    )


def liquid_chemical_potential(n, params, temp):
    """df/dn of the uniform state: (beta q0^4 + eps) n + g n^2 + v0 n^3 (no entropic +1)."""
    p = validate_params(params)
    beta, g, v0, q0 = p['beta'], p['g'], p['v0'], p['q0']
    return (
        (beta*q0**4 + temp)*n +
        g*n**2 +
        v0*n**3
    )


def _band(g, v0, eps_eff, width):
    # n_c -/+ width * sqrt(g^2 - 3 v0 eps_eff) / v0, or None where the root is not real
    disc = g**2 - 3*v0*eps_eff
    if not disc > 0:
        return None
    n_c = -g/(3*v0)
    half = width*np.sqrt(disc)/v0
    return (n_c - half, n_c + half)


def critical_point(params):
    """(n_c, eps_c) = (-g/(3 v0), g^2/(3 v0)): minimum of h(n) = 2 g n + 3 v0 n^2, eps_c = -h(n_c)."""
    p = validate_params(params)
    g, v0 = p['g'], p['v0']
    return -g/(3*v0), g**2/(3*v0)


def spinodal(params, temp):
    """k = q0 linear-instability band: eps + 2 g n + 3 v0 n^2 < 0, i.e.
    n_c -/+ sqrt(g^2 - 3 v0 eps)/(3 v0); None at and above eps_c. Independent of beta and q0."""
    p = validate_params(params)
    return _band(p['g'], p['v0'], temp, 1.0/3)


def liquid_spinodal(params, temp):
    """k = 0 band where the uniform branch is concave, beta q0^4 + eps + 2 g n + 3 v0 n^2 < 0, or None."""
    p = validate_params(params)
    return _band(p['g'], p['v0'], p['beta']*p['q0']**4 + temp, 1.0/3)


def existence_bounds(params, temp):
    """One-mode existence intervals per structure, or None at and above eps_c.

    eps_SN(n) = -h(n) + f3^2/(10 f4) with h = 2 g n + 3 v0 n^2, f3 = 2 g + 6 v0 n,
    f4 = 6 v0 (the log adapter's formula for this free energy) is a quadratic in n
    with roots n_c -/+ sqrt(5 (g^2 - 3 v0 eps))/(6 v0). Triangle from the lower root
    to n_plus, honeycomb from n_minus to the upper root, stripes on [n_minus, n_plus]."""
    p = validate_params(params)
    band = spinodal(p, temp)
    if band is None:
        return None
    sn = _band(p['g'], p['v0'], temp, np.sqrt(5.0)/6)
    n_minus, n_plus = band
    return {
        'triangle': (sn[0], n_plus),
        'stripe': (n_minus, n_plus),
        'honeycomb': (n_minus, sn[1]),
    }


def scan_window(structure, params, temp, pad=0.02):
    """Suggested density window for a q0* scan: the one-mode existence interval of
    `structure` widened by `pad` on each side (no positivity clamp); None at and
    above eps_c."""
    bounds = existence_bounds(params, temp)
    if bounds is None:
        return None
    lo, hi = bounds[_structure(structure).name]
    return (lo - pad, hi + pad)
