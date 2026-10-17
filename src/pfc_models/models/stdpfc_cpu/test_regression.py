"""Regression and physics tests for stdpfc_cpu.

`test_free_energy_at_fixed_step` is the regression proper: a pinned configuration
and step, asserted against a recorded reference. The rest check this model's
physics against closed forms, so they assert arithmetic rather than a recorded
number and cannot go stale:

* `test_free_energy_single_mode` - every term of f integrates exactly for
  `n = n_bar + 2a cos(kx)`, including the gradient term via Parseval. Pins the
  spectral operator, the FFT normalisation and all four polynomial coefficients.
  Measured error: 0.0.
* `test_etd_is_exact_in_the_linear_regime` - ETD1 integrates the linear operator
  exactly, so a mode at amplitude 1e-6 must grow as exp(L t) to roundoff. Pins
  Gamma, beta, q0, epsilon, the Laplacian sign and both ETD coefficients.
  Measured relative error: 6e-14 to 2e-13 over 100 steps.
* `test_cubic_removal_is_exact_at_fixed_field` and
  `test_cubic_removal_converges_as_dt_shrinks` - the reason `g` is still a
  parameter. See the block comment above them.
* `test_numba_parallel_matches_serial` - `use_parallel` selects a code path by
  grid size; both paths must compute the same thing.

Numeric stability (relaxation, mass conservation) is not here: that is one test
per model in `tests/test_smoke_models.py`, which is where every model's
stability statement lives.

The regression configuration, all of it load-bearing - change any line and the
reference is void:

The point is not accuracy - it is detecting the day a change silently alters the
physics. Everything here is deterministic: the perfect-crystal-with-central-vacancy
generator draws no randomness, so no seed or RNG convention is involved.

Configuration (all of it load-bearing - change any line and the reference is void):

    Nx = Ny = 132 on Lx = 14 a, Ly = 8 h   (a = 4 pi / sqrt(3), h = 4 pi, q0 = 1)
    beta = (3/4)^(1/2) * 0.5               epsilon = (3/4)^(1/2) * (-0.9 + 2*(-2) + 4)
    g    = (3/4)^(3/4) * (-2 + 5/2)        mean n  = (3/4)^(1/4) * (0.55 - 1)
    vacancy mean = (3/4)^(1/4) * (0.30 - 1)    A0 = (3/4)^(1/4) * 0.19
    dt = 0.1, 200 steps

The state point is the alpha = 1 image of the logpfc example notebook's, so this
test and the log family's run at corresponding points of the two branches. The
vacancy keeps the run off its fixed point: the free energy is still moving in the
fifth decimal between step 200 and step 400, so the assertion is sensitive rather
than pinned to an equilibrium the model would reach regardless.

Reference recorded 2026-10-16 from this implementation on pyfftw 0.15.1 / numba 0.68.0,
NumPy 2.5.4, float64. It is reproducible to 1 ulp (1.4e-17 absolute) across
nThreads = 1, 2, 4 and 12 and across repeat processes, so RTOL = 1e-10 leaves six
orders of margin; it is not a tolerance for a different backend, which is why
stdpfc carries its own.

Re-record with:

    python -m pfc_models.models.stdpfc_cpu.test_regression
"""
import numpy as np
import pytest

pytest.importorskip("pyfftw", reason="CPU runtime only")
pytest.importorskip("numba", reason="CPU runtime only")


REFERENCE_FREE_ENERGY = -0.11237092135766821
REFERENCE_STEP = 200
RTOL = 1e-10

Q0 = 1.0
A_LAT = 4 * np.pi / 3**0.5 / Q0
H_LAT = 4 * np.pi / Q0
NX = NY = 132
LX = 14 * A_LAT
LY = 8 * H_LAT
DT = 0.1

C12, C34, C14 = (3/4)**0.5, (3/4)**0.75, (3/4)**0.25
BETA = C12 * 0.5
EPSILON = C12 * (-0.9 + 2*(-2.0) + 4)
G = C34 * (-2.0 + 2.5)
MEAN_DENSITY_SOLID = C14 * (0.55 - 1)
MEAN_DENSITY_VACANCY = C14 * (0.30 - 1)
AMPLITUDE = C14 * 0.19

MASS_TOL = 1e-12


def _run_to_reference_step():
    """Build the pinned configuration, step to REFERENCE_STEP, return (free energy, mass drift)."""
    from pfc_models.fields.initial_conditions_2d import (
        GeneratePerfectCrystalInitialWithCentralVacancy,
    )
    from pfc_models.models.stdpfc_cpu import simulation as sim_mod

    geometry = sim_mod.Build2DGeometry(Nx=NX, Ny=NY, Lx=LX, Ly=LY)
    params = sim_mod.BuildStdPFCModelParameters(
        Gamma=1.0, beta=BETA, q0=Q0, epsilon=EPSILON, g=G,
        dt=DT, meanDensity=MEAN_DENSITY_SOLID,
    )
    sim = sim_mod.BuildNumpyCPU_Simulation(params, geometry, nThreads=4)
    initial = GeneratePerfectCrystalInitialWithCentralVacancy(
        geometry, MEAN_DENSITY_SOLID, MEAN_DENSITY_VACANCY, A0=AMPLITUDE, q0=Q0)
    sim.State.n[...] = initial
    mass_start = float(sim.State.n.mean())
    for _ in range(REFERENCE_STEP):
        sim.TimeStep()
    return float(sim.CalculateFreeEnergy()), abs(float(sim.State.n.mean()) - mass_start)


def test_free_energy_at_fixed_step():
    free_energy, mass_drift = _run_to_reference_step()
    assert mass_drift < MASS_TOL, f"mass not conserved: drift {mass_drift:.3e}"
    assert free_energy == pytest.approx(REFERENCE_FREE_ENERGY, rel=RTOL), (
        f"free energy at step {REFERENCE_STEP} is {free_energy!r}, "
        f"reference {REFERENCE_FREE_ENERGY!r} (rtol {RTOL})"
    )


# --------------------------------------------------------------------------
# the free energy and the time stepper, against closed forms
# --------------------------------------------------------------------------
# The two tests below use a square box and a coarse grid. The mode must sit on
# the grid and its fourth harmonic must stay below Nyquist, or n**4 aliases and
# the comparison stops being exact: mode 4 on 64 points gives harmonics up to 16
# against a Nyquist of 32.
MODE_NX = 64
MODE_L = 28 * A_LAT
MODE_STEPS = 100


def _build(beta=None, epsilon=None, g=None, mean=None, dt=DT,
           nx=None, lx=None, ly=None):
    """Build a simulation, defaulting to the regression configuration."""
    from pfc_models.models.stdpfc_cpu import simulation as sim_mod

    geometry = sim_mod.Build2DGeometry(
        Nx=nx if nx is not None else NX,
        Ny=nx if nx is not None else NY,
        Lx=lx if lx is not None else LX,
        Ly=ly if ly is not None else LY,
    )
    params = sim_mod.BuildStdPFCModelParameters(
        Gamma=1.0,
        beta=BETA if beta is None else beta,
        q0=Q0,
        epsilon=EPSILON if epsilon is None else epsilon,
        g=G if g is None else g,
        dt=dt,
        meanDensity=MEAN_DENSITY_SOLID if mean is None else mean,
    )
    return sim_mod.BuildNumpyCPU_Simulation(params, geometry, nThreads=4), geometry


def test_free_energy_single_mode():
    """f for n = n_bar + 2a cos(kx) is exact in closed form; the code must match it.

    <n (q0^2 + lap)^2 n> = n_bar^2 q0^4 + 2 a^2 (q0^2 - k^2)^2  (Parseval),
    <n^2> = n_bar^2 + 2a^2, <n^3> = n_bar^3 + 6 n_bar a^2,
    <n^4> = n_bar^4 + 12 n_bar^2 a^2 + 6 a^4.
    """
    n_bar, amp, mode = -0.3, 0.07, 4
    sim, geometry = _build(nx=MODE_NX, lx=MODE_L, ly=MODE_L, mean=0.0)
    k = 2 * np.pi * mode / MODE_L
    sim.State.n[...] = n_bar + 2 * amp * np.cos(k * geometry.X)

    expected = (
        (BETA / 2) * (n_bar**2 * Q0**4 + 2 * amp**2 * (Q0**2 - k**2)**2)
        + (EPSILON / 2) * (n_bar**2 + 2 * amp**2)
        + (G / 3) * (n_bar**3 + 6 * n_bar * amp**2)
        + 0.25 * (n_bar**4 + 12 * n_bar**2 * amp**2 + 6 * amp**4)
    )
    got = float(sim.CalculateFreeEnergy())
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-14), (
        f"free energy {got!r} != closed form {expected!r}"
    )


@pytest.mark.parametrize("mode", [3, 4, 5])
def test_etd_is_exact_in_the_linear_regime(mode):
    """A tiny single mode must grow as exp(L t) with L the model's linear operator.

    ETD1 integrates the linear part exactly, so at amplitude 1e-6 - where
    g n**2 + n**3 is ~1e-12 relative - the only error is roundoff.
    """
    amp = 1e-6
    sim, geometry = _build(nx=MODE_NX, lx=MODE_L, ly=MODE_L, mean=0.0)
    k = 2 * np.pi * mode / MODE_L
    sim.State.n[...] = amp * np.cos(k * geometry.X)

    k_lap = -k**2
    linear = 1.0 * k_lap * (BETA * (Q0**2 + k_lap)**2 + EPSILON)   # Gamma = 1
    for _ in range(MODE_STEPS):
        sim.TimeStep()

    spectrum = np.fft.rfft2(sim.State.n)
    got = float(np.abs(spectrum[0, mode])) / (MODE_NX * MODE_NX / 2)
    expected = amp * np.exp(linear * DT * MODE_STEPS)
    assert got == pytest.approx(expected, rel=1e-8), (
        f"mode {mode}: amplitude {got:.9e} != exp(L t) {expected:.9e}"
    )


# --------------------------------------------------------------------------
# removability of the cubic term - the reason g is still a parameter
# --------------------------------------------------------------------------
# g is removable by n -> n + g/3 with epsilon -> epsilon - g**2/3 (derivation
# sections 2.2-2.3), and the two models generate identical dynamics *in
# continuous time*. They do not generate identical trajectories under ETD1, and
# that is not a bug: the shift moves g**2/3 out of the nonlinear term and into
# the linear operator, so the two runs split linear from nonlinear differently
# and their first-order splitting errors differ. Measured at Nx = 132, T = 2,
# crystal IC:
#
#     dt = 0.1      max|n_B - (n_A + g/3)| = 2.56e-04
#     dt = 0.05                              1.33e-04   ratio 1.92
#     dt = 0.025                             6.78e-05   ratio 1.96
#     dt = 0.0125                            3.42e-05   ratio 1.98
#     dt = 0.00625                           1.72e-05   ratio 1.99
#
# First order, as expected. So the trajectory test asserts the convergence rate,
# not agreement at any one dt, and the exact statement is tested at fixed field.
# Asserting pointwise agreement at dt = 0.1 would be asserting something the
# time stepper does not promise.


def _removal_offsets(mean_density):
    """The two constants the shift generates, from the derivation's algebra."""
    s = G / 3
    energy = (BETA * Q0**4 * s * (mean_density + s / 2)
              + (EPSILON * s - 2 * s**3) * mean_density
              + 0.5 * EPSILON * s**2 - 1.25 * s**4)
    potential = EPSILON * s - 2 * s**3 + BETA * Q0**4 * s
    return s, energy, potential


def test_cubic_removal_is_exact_at_fixed_field():
    """At one field: mu differs by a constant, and <f> by the closed-form offset.

    mu_B - mu_A is spatially constant, so the Laplacian in front of mu
    annihilates it and the two models have the same dynamics; <f_B> - <f_A> is
    fixed because the shift generates only linear and constant terms and the
    mean is conserved.
    """
    sim_a, geometry = _build(nx=MODE_NX, lx=MODE_L, ly=MODE_L, mean=0.0)
    rng = np.random.default_rng(7)
    field = MEAN_DENSITY_SOLID + 0.3 * np.fft.irfft2(
        np.fft.rfft2(rng.normal(size=geometry.X.shape))
        * np.exp(-(geometry.KX**2 + geometry.KY**2)), s=geometry.X.shape)
    mean_density = float(field.mean())
    s, energy_offset, potential_offset = _removal_offsets(mean_density)

    sim_b, _ = _build(nx=MODE_NX, lx=MODE_L, ly=MODE_L,
                      epsilon=EPSILON - G**2/3, g=0.0, mean=mean_density + s)
    sim_a.State.n[...] = field
    sim_b.State.n[...] = field + s

    delta_f = float(sim_b.CalculateFreeEnergy()) - float(sim_a.CalculateFreeEnergy())
    assert delta_f == pytest.approx(energy_offset, rel=1e-10, abs=1e-15), (
        f"<f_B> - <f_A> = {delta_f!r}, closed form {energy_offset!r}"
    )

    for sim in (sim_a, sim_b):
        sim._Batch1_CalcNonlinMu()
        sim._Batch2_CalcMu_Terms()
    delta_mu = sim_b.State.mu - sim_a.State.mu
    assert np.ptp(delta_mu) < 1e-11, (
        f"mu_B - mu_A is not constant: spread {np.ptp(delta_mu):.3e}"
    )
    assert float(delta_mu.mean()) == pytest.approx(potential_offset, rel=1e-9), (
        f"mu offset {delta_mu.mean()!r} != predicted {potential_offset!r}"
    )


def test_cubic_removal_converges_as_dt_shrinks():
    """The paired trajectories converge at first order in dt. See the block comment."""
    from pfc_models.fields.initial_conditions_2d import (
        GeneratePerfectCrystalInitialConditions,
    )
    total_t = 2.0
    s = G / 3

    gaps = []
    for dt in (0.1, 0.05, 0.025):
        sim_a, geometry = _build(dt=dt)
        initial = GeneratePerfectCrystalInitialConditions(
            geometry, MEAN_DENSITY_SOLID, A0=AMPLITUDE, q0=Q0)
        sim_b, _ = _build(dt=dt, epsilon=EPSILON - G**2/3, g=0.0,
                          mean=MEAN_DENSITY_SOLID + s)
        sim_a.State.n[...] = initial
        sim_b.State.n[...] = initial + s

        for _ in range(int(round(total_t / dt))):
            sim_a.TimeStep()
            sim_b.TimeStep()
        gaps.append(float(np.abs(sim_b.State.n - (sim_a.State.n + s)).max()))

    ratios = [gaps[i] / gaps[i + 1] for i in range(len(gaps) - 1)]
    assert all(r > 1.8 for r in ratios), (
        f"gap is not first order in dt: gaps {gaps}, ratios {ratios}"
    )
    assert gaps[-1] < 1e-4, f"gap at the finest dt is {gaps[-1]:.3e}, expected < 1e-4"


# --------------------------------------------------------------------------
# CPU only: the two Numba code paths must agree
# --------------------------------------------------------------------------
def test_numba_parallel_matches_serial():
    """`use_parallel` picks a code path by grid size; both must compute the same thing."""
    from pfc_models.models.stdpfc_cpu import simulation as sim_mod

    rng = np.random.default_rng(0)
    field = rng.normal(size=(64, 64))
    parallel, serial = np.empty_like(field), np.empty_like(field)
    sim_mod._nonlin_mu_parallel(field, parallel, G)
    sim_mod._nonlin_mu_serial(field, serial, G)
    assert np.array_equal(parallel, serial), "nonlin_mu paths disagree"

    shape = (64, 33)
    out_parallel = np.empty(shape, dtype=complex)
    out_serial = np.empty(shape, dtype=complex)
    k_linexp = rng.normal(size=shape)
    k_nonlinexp = rng.normal(size=shape)
    n_hat = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    k_lap = rng.normal(size=shape)
    nonlin_mu_hat = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    sim_mod._n_next_parallel(out_parallel, k_linexp, k_nonlinexp, n_hat, k_lap, nonlin_mu_hat, 1.0)
    sim_mod._n_next_serial(out_serial, k_linexp, k_nonlinexp, n_hat, k_lap, nonlin_mu_hat, 1.0)
    assert np.array_equal(out_parallel, out_serial), "n_next paths disagree"


if __name__ == "__main__":
    value, drift = _run_to_reference_step()
    print(f"REFERENCE_FREE_ENERGY = {value!r}   # mass drift {drift:.3e}")
