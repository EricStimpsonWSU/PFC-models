"""Phase-diagram adapter for logpfc_cpu: the model side of pfc_models.phase_diagram.

A module, not a class: every supporting model provides these names (design note
sec. 2); siblings repeat the contract and `tools/variant_check.py` watches the
duplication.

    SLUG, STRUCTURES, validate_params, validate_cell, validate_ansatz
    build, get_state, set_state, get_time, set_time, get_dt, step, free_energy
    diagnose
    liquid_free_energy, liquid_chemical_potential
    spinodal, critical_point, liquid_spinodal, existence_bounds, scan_window

Model: f = phi beta/2 (q0^2 + lap)^2 phi + phi ln phi + eps/2 phi^2 + g/3 phi^3
+ v0/4 phi^4 with v0 = 1 hard-wired in `simulation.py` (so `validate_params`
rejects any other v0). `temp` is epsilon; `meanDensity` is phibar (a float).

Structures and cell. The three R2C 3-mode sets of the legacy `lattice_vals`
(triangle, honeycomb, stripe); the C2C sets are not carried. k-vector entries are
(row, col) = (ky, kx) indices for one rectangular unit cell, scaled by (My, Mx)
for an Mx x My supercell. The box is Lx = Mx a, Ly = My h with
a = 4 pi / (sqrt(3) q00) and h = 4 pi / q00, grid (uNx Mx, uNy My).

Positive-density assumptions live here and only here: the ansatz amplitude is
amplitudeSign * amplitudeMax * phibar * A_i (amplitudeMax is the one-mode
positivity bound in units of phibar), a field with a negative minimum is rescaled
about phibar so its minimum is zero, and the fitness amplitudes are normalised by
phibar.

Provenance: `_build3Mode`, `_calcFitness`, `_buildModel` and
`_buildSystemGeometry` of warrior/phase-diagram/_minimize_f.py
(`MinimizeFreeEnergy`), relocated with their arithmetic unchanged so the
minimiser's results stay bit-identical; the builder is this model's
`BuildCupyGPU_Simulation` (the CPU FFTW builder, P02-01 sec. 2.1). Two
differences, neither numerical: `diagnose` refreshes `State.phi_hat` itself (the
original relied on `CalculateFreeEnergy` having just done so, with the same
FFTW call), and it adds a maxima-per-cell label to `details`. `irfftn` gets
`axes=(0,1)`, the axes NumPy already used for `axes=None` (NumPy 2 deprecates
omitting them when `s` is given).
The analytic bounds are those of the hand-off addendum (31 Oct 2026).
"""
from __future__ import annotations

from dataclasses import dataclass

# numerics
import numpy as np
import pyfftw
from scipy.optimize import brentq

from pfc_models.models.logpfc_cpu import simulation as _simulation
from pfc_models.phase_diagram.q0star import Diagnostics

SLUG = 'logpfc_cpu'


@dataclass(frozen=True)
class Structure:
    """One target lattice. `kvectors` = (1-mode, 2-mode, 3-mode) sets of R2C
    (row, col) indices for one rectangular cell; `maxima_per_cell` is the expected
    count of density maxima per rectangular cell (None: ridges, not points)."""
    name: str
    kvectors: tuple
    amplitudeSign: int
    amplitudeMax: float
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
        amplitudeMax=1.0/3,
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
        amplitudeMax=1.0/6,
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
        amplitudeMax=1.0/2,
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
    """mode (label, e.g. 'L2H', 'H2L'); A1, A2, A3: 1-, 2-, 3-mode amplitudes in units of amplitudeMax."""
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


# Build 3-mode state for initial conditions and fitness evaluation
def _build3Mode(geometry, structure, cell, meanDensity, A, B, C):
    X, Y = geometry.X, geometry.Y
    Nx, Ny = cell['uNx']*cell['Mx'], cell['uNy']*cell['My']
    (rows_1mode, cols_1mode), (rows_2mode, cols_2mode), (rows_3mode, cols_3mode) = _mode_indices(structure, cell)

    # one-mode ansatz
    n_H_hat = np.zeros_like(X, dtype=complex)
    n_H_hat[0,0] = meanDensity*X.size
    n_H_hat[rows_1mode, cols_1mode] = A*X.size
    n_H_hat[rows_2mode, cols_2mode] = B*X.size
    n_H_hat[rows_3mode, cols_3mode] = C*X.size

    n_H = np.fft.irfftn(n_H_hat, s=(Ny,Nx), axes=(0,1)).real

    # rescale to avoid negative densities
    minDensity = n_H.min()
    if minDensity < 0:
        n_H = meanDensity + (n_H - meanDensity) * meanDensity / (meanDensity - minDensity)
    return n_H


def _refresh_phi_hat(sim):
    # the FFT CalculateFreeEnergy performs, so diagnose does not depend on call order
    phi_copy = pyfftw.byte_align(np.copy(sim.State.phi))
    sim.fft_fwd(phi_copy, sim.State.phi_hat)
    return sim.State.phi_hat


# Calculate fitness score: 1 - L2 Norm of density vs 3-mode approximation [1 = Perfect match, << 1 Poor match (no minimum)]
def _calcFitness(sim, structure, cell):
    phi = sim.phi
    phi_hat = _refresh_phi_hat(sim)
    Nx, Ny = cell['uNx']*cell['Mx'], cell['uNy']*cell['My']
    simSize = Nx * Ny
    meanDensity = phi_hat.flat[0]/simSize
    (rows_1mode, cols_1mode), (rows_2mode, cols_2mode), (rows_3mode, cols_3mode) = _mode_indices(structure, cell)

    A = structure.amplitudeSign * np.nanmean(np.abs(phi_hat[rows_1mode, cols_1mode])) / simSize / meanDensity
    B = structure.amplitudeSign * np.nanmean(np.abs(phi_hat[rows_2mode, cols_2mode])) / simSize / meanDensity
    C = structure.amplitudeSign * np.nanmean(np.abs(phi_hat[rows_3mode, cols_3mode])) / simSize / meanDensity
    phi_3mode = _build3Mode(sim.Geometry, structure, cell, meanDensity, A*meanDensity, B*meanDensity, C*meanDensity)
    return 1 - float(np.abs(np.power(phi - phi_3mode, 2).mean())), np.abs(A), np.abs(B), np.abs(C)


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

    # build model
    model = _simulation.BuildLogPFCModelParameters(
        Gamma = params['Gamma'],
        beta = params['beta'],
        q0 = params['q0'],
        epsilon = temp,
        g = params['g'],
        dt = dt,
        meanDensity = meanDensity
    )

    # build geometry
    geometry = _box(cell, q00)

    # Initialize density field
    if state is None:
        n_H = _build3Mode(
            geometry, structure, cell,
            meanDensity,
            structure.amplitudeSign*structure.amplitudeMax*meanDensity*ansatz['A1'],
            structure.amplitudeSign*structure.amplitudeMax*meanDensity*ansatz['A2'],
            structure.amplitudeSign*structure.amplitudeMax*meanDensity*ansatz['A3']
        )
    else:
        n_H = state

    # Build simulation
    sim = _simulation.BuildCupyGPU_Simulation(
        modelParameters = model,
        geometry = geometry,
        nThreads = nThreads
    )
    sim.State.phi[...] = n_H
    return sim


def get_state(sim):
    return sim.phi.copy()


def set_state(sim, state):
    sim.phi[...] = state


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
    """fitness = 1 - mean((phi - phi_3mode)^2); amplitudes (A, B, C) = mean |phi_hat| over
    the 1-, 2-, 3-mode sets in units of phibar; details: maxima_per_cell."""
    structure = _structure(structure)
    cell = validate_cell(cell)
    fitness, A, B, C = _calcFitness(sim, structure, cell)
    return Diagnostics(
        fitness=fitness,
        amplitudes=(A, B, C),
        details={'maxima_per_cell': _maxima_per_cell(sim.phi, cell)},
    )


# ---------------------------------------------------------------- uniform state, analytic

def liquid_free_energy(phi, params, temp):
    """Uniform f: phi ln phi + (beta q0^4 + eps)/2 phi^2 + g/3 phi^3 + v0/4 phi^4."""
    p = validate_params(params)
    beta, g, v0, q0 = p['beta'], p['g'], p['v0'], p['q0']
    return (
        phi*np.log(phi) +
        1./2*(beta*q0**4 + temp)*phi**2 +
        1./3*g*phi**3 +
        1./4*v0*phi**4
    )


def liquid_chemical_potential(phi, params, temp):
    """df/dphi of the uniform state, including the +1 that the simulation's
    nonlinear mu (ln phi + g phi^2 + phi^3) omits because grad^2 annihilates it."""
    p = validate_params(params)
    beta, g, v0, q0 = p['beta'], p['g'], p['v0'], p['q0']
    return (
        1 + np.log(phi) +
        (beta*q0**4 + temp)*phi +
        g*phi**2 +
        v0*phi**3
    )


def _h(phi, g, v0):
    # f''(phi) of the local part minus eps: 1/phi + 2 g phi + 3 v0 phi^2
    return 1/phi + 2*g*phi + 3*v0*phi**2


def _f3(phi, g, v0):
    return -1/phi**2 + 2*g + 6*v0*phi


def _f4(phi, g, v0):
    return 2/phi**3 + 6*v0


def _phi_c(g, v0):
    # h'(phi) = f3 is increasing on phi > 0 (f4 > 0) and has a single root
    hi = 1.0
    while _f3(hi, g, v0) <= 0:
        hi *= 2
    return brentq(lambda x: _f3(x, g, v0), 1e-8, hi, xtol=1e-15, maxiter=500)


def _convex_band(F, phi_c):
    """Roots of a convex F on (0, inf) either side of its minimiser phi_c, or None if F(phi_c) >= 0."""
    if not F(phi_c) < 0:
        return None
    lo = phi_c
    while F(lo) < 0:
        lo /= 2
    hi = phi_c
    while F(hi) < 0:
        hi *= 2
    return (brentq(F, lo, phi_c, xtol=1e-15, maxiter=500),
            brentq(F, phi_c, hi, xtol=1e-15, maxiter=500))


def critical_point(params):
    """(phi_c, eps_c): 6 v0 phi^3 + 2 g phi^2 - 1 = 0, eps_c = -h(phi_c)."""
    p = validate_params(params)
    g, v0 = p['g'], p['v0']
    phi_c = _phi_c(g, v0)
    return phi_c, -_h(phi_c, g, v0)


def spinodal(params, temp):
    """k = q0 linear-instability band (phi_minus, phi_plus), or None at and above eps_c.

    Positive roots of 3 v0 phi^3 + 2 g phi^2 + eps phi + 1 = 0, i.e. eps + h(phi) = 0;
    independent of beta and q0."""
    p = validate_params(params)
    g, v0 = p['g'], p['v0']
    return _convex_band(lambda x: temp + _h(x, g, v0), _phi_c(g, v0))


def liquid_spinodal(params, temp):
    """k = 0 band where the uniform branch is concave, beta q0^4 + eps + h(phi) < 0, or None."""
    p = validate_params(params)
    beta, g, v0, q0 = p['beta'], p['g'], p['v0'], p['q0']
    return _convex_band(lambda x: beta*q0**4 + temp + _h(x, g, v0), _phi_c(g, v0))


def _eps_SN(phi, g, v0):
    # one-mode Landau saddle-node: <s^2> = 6, <s^3> = +-12, <s^4> = 90
    return -_h(phi, g, v0) + 0.1*_f3(phi, g, v0)**2/_f4(phi, g, v0)


def _sn_root(G, start, stop, n=4000):
    """First sign change of G walking from start (G > 0) towards stop, refined by brentq."""
    grid = np.geomspace(start, stop, n)
    vals = np.array([G(x) for x in grid])
    k = np.flatnonzero(vals <= 0)
    if k.size == 0:
        return None
    k = k[0]
    return brentq(G, grid[k-1], grid[k], xtol=1e-15, maxiter=500)


def existence_bounds(params, temp):
    """One-mode existence intervals per structure, or None at and above eps_c.

    eps_SN(phi) = -h(phi) + f3(phi)^2 / (10 f4(phi)), with f3 and f4 the third and
    fourth derivatives of the local free energy (hand-off addendum). The triangle
    exists from the root below phi_c up to phi_plus, the honeycomb from phi_minus
    up to the root above phi_c, stripes (supercritical in one-mode) on
    [phi_minus, phi_plus]."""
    p = validate_params(params)
    g, v0 = p['g'], p['v0']
    band = spinodal(p, temp)
    if band is None:
        return None
    phi_c = _phi_c(g, v0)
    G = lambda x: _eps_SN(x, g, v0) - temp
    phi_sn_tri = _sn_root(G, phi_c, phi_c*1e-4)
    phi_sn_hc = _sn_root(G, phi_c, phi_c*1e4)
    phi_minus, phi_plus = band
    return {
        'triangle': (phi_sn_tri, phi_plus),
        'stripe': (phi_minus, phi_plus),
        'honeycomb': (phi_minus, phi_sn_hc),
    }


def scan_window(structure, params, temp, pad=0.02):
    """Suggested density window for a q0* scan: the one-mode existence interval of
    `structure` (addendum table: triangle [phi_SN_tri, ~phi_plus], stripe
    [phi_minus, phi_plus], honeycomb [~phi_minus, phi_SN_hc]) widened by `pad`
    on each side (the lower end never below half the interval's lower bound);
    None at and above eps_c. The one-mode estimates degrade below eps ~ -1."""
    bounds = existence_bounds(params, temp)
    if bounds is None:
        return None
    lo, hi = bounds[_structure(structure).name]
    return (max(lo - pad, 0.5*lo), hi + pad)
