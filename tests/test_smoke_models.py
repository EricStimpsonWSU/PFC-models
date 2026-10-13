"""Smoke tests: each migrated model builds, steps, conserves mass, and relaxes.

Not a regression test - no reference free energy is asserted, by decision for
this migration step. What this checks is that the model integrates: that the
substrate wiring is correct, that mass is conserved to machine precision, and
that the free energy goes downhill, which is the one qualitative statement
conserved gradient dynamics owes us.

Cost and parameter choices, all measured rather than assumed
-----------------------------------------------------------
`DT = 0.1`, `NSTEPS = 100` (so totalT = 10), against the example notebooks'
`totalT = 10000` at the same dt - 100_000 steps, which took 9m42s for
logpfc_cpu under nbmake. This runs in about 9 seconds.

`NX = NY = 66 * 4` is the notebooks' grid and is NOT negotiable downward.
The box is 28 lattice spacings by 16 rows, so resolution per lattice spacing is:

    Nx = 264  ->  9.4 points/a   free energy strictly decreasing
    Nx = 132  ->  4.7 points/a   NOT monotonic: 8 of 100 steps increase,
                                 largest +2.7e-05
    Nx =  66  ->  2.4 points/a   monotonic, but the crystal is unresolved

Shrinking the grid to make the test faster therefore breaks the assertion for
numerical reasons. Measured on logpfc_cpu with pyfftw 0.15.1 / numba 0.68.0.

The initial condition is the deterministic perfect crystal with a central
vacancy, so no seed is involved and the test does not depend on the RNG
convention.

Monotonicity and logpfc_extv
----------------------------
`test_logpfc_extv_runs` deliberately does NOT assert per-step monotonic
decrease. Two reasons, both in the model as migrated:

* `ValidatePhiRange` runs every step and calls `OptimizeM`, which recomputes
  `M` and rebuilds the linear mu kernel. The operator - and so the functional
  being minimised - changes during the run.
* `CalculateFreeEnergy` is inherited unchanged from the base forks: it sums the
  gradient, ideal-gas and polynomial terms only. Neither the external-potential
  coupling nor the `-M*phi` term contributes, so the number it returns is not
  the Lyapunov functional of its own dynamics.

Asserting monotonicity there would be asserting something the physics does not
promise. It gets the finiteness, positivity and mass-conservation checks
instead.
"""
import numpy as np
import pytest

DT = 0.1
NSTEPS = 100            # totalT = DT * NSTEPS = 10
NX = NY = 66 * 4

Q0 = 1.0
A_LAT = 4 * np.pi / 3**0.5 / Q0     # triangular lattice spacing
H_LAT = 4 * np.pi / Q0              # row spacing
LX = 7 * 4 * A_LAT
LY = 4 * 4 * H_LAT

MEAN_DENSITY_SOLID = 0.55
MEAN_DENSITY_VACANCY = 0.3
AMPLITUDE = 0.15

# mass must hold to machine precision: the k=0 mode of the update is pinned
MASS_TOL = 1e-12


def _seed_crystal(sim, geometry):
    """Fill sim.State.phi with a perfect crystal holding one central vacancy.

    The generators in pfc_models.fields are NumPy-only by design - they import
    numpy alone and return npFloatType - so a GPU model needs an explicit host
    to device transfer. This mirrors what the example notebooks do:

        sim.State.phi[...] = cp.asarray(phiInit, dtype=cpFloatType)

    on CuPy models, and a plain assignment on the NumPy/FFTW one. Note that
    CuPy will NOT transfer implicitly here: `cupy_array[...] = numpy_array`
    raises "non-scalar numpy.ndarray cannot be used for fill" rather than
    copying. Keyed off the destination buffer so one helper serves all three
    models; dtype comes from that buffer, so float32 and float64 both work.
    """
    from pfc_models.fields.initial_conditions_2d import (
        GeneratePerfectCrystalInitialWithCentralVacancy,
    )
    phi0 = GeneratePerfectCrystalInitialWithCentralVacancy(
        geometry, MEAN_DENSITY_SOLID, MEAN_DENSITY_VACANCY, A0=AMPLITUDE, q0=Q0
    )
    target = sim.State.phi
    if type(target).__module__.startswith("cupy"):
        import cupy as cp
        phi0 = cp.asarray(phi0, dtype=target.dtype)
    target[...] = phi0


def _march(sim, nsteps=NSTEPS):
    """Step the model, sampling free energy and mean density every step."""
    free_energy = [float(sim.CalculateFreeEnergy())]
    mass = [float(sim.State.phi.mean())]
    for _ in range(nsteps):
        sim.TimeStep()
        free_energy.append(float(sim.CalculateFreeEnergy()))
        mass.append(float(sim.State.phi.mean()))
    return np.array(free_energy), np.array(mass)


def _assert_sane(free_energy, mass, phi_min):
    assert np.isfinite(free_energy).all(), "free energy went non-finite"
    assert np.isfinite(mass).all(), "mean density went non-finite"
    assert phi_min > 0.0, f"phi went non-positive (min {phi_min:.3e})"
    drift = float(mass.max() - mass.min())
    assert drift < MASS_TOL, f"mass not conserved: drift {drift:.3e}"


def _assert_relaxes(free_energy):
    """Conserved gradient dynamics: the free energy must go downhill."""
    deltas = np.diff(free_energy)
    rising = np.flatnonzero(deltas >= 0.0)
    assert rising.size == 0, (
        f"free energy increased at {rising.size} of {deltas.size} steps; "
        f"largest increase {deltas.max():+.3e}, first at step {int(rising[0]) if rising.size else -1}"
    )
    assert free_energy[-1] < free_energy[0], "no net relaxation over the run"


def test_logpfc_cpu_relaxes():
    """log-PFC on NumPy + PyFFTW + Numba."""
    pytest.importorskip("pyfftw", reason="CPU runtime only")
    pytest.importorskip("numba", reason="CPU runtime only")
    from pfc_models.models.logpfc_cpu import simulation as sim_mod

    geometry = sim_mod.Build2DGeometry(Nx=NX, Ny=NY, Lx=LX, Ly=LY)
    params = sim_mod.BuildLogPFCModelParameters(
        Gamma=1.0, beta=0.5, q0=Q0, epsilon=-0.9, g=-2.0,
        dt=DT, meanDensity=MEAN_DENSITY_SOLID,
    )
    sim = sim_mod.BuildCupyGPU_Simulation(params, geometry, nThreads=4)
    _seed_crystal(sim, geometry)

    free_energy, mass = _march(sim)
    _assert_sane(free_energy, mass, float(sim.State.phi.min()))
    _assert_relaxes(free_energy)


def test_logpfc_relaxes():
    """log-PFC on CuPy. Same scheme as logpfc_cpu; only the array module differs."""
    pytest.importorskip("cupy", reason="GPU runtime only")
    from pfc_models.models.logpfc import simulation as sim_mod

    geometry = sim_mod.Build2DGeometry(Nx=NX, Ny=NY, Lx=LX, Ly=LY)
    params = sim_mod.BuildLogPFCModelParameters(
        Gamma=1.0, beta=0.5, q0=Q0, epsilon=-0.9, g=-2.0,
        dt=DT, meanDensity=MEAN_DENSITY_SOLID,
    )
    sim = sim_mod.BuildCupyGPU_Simulation(params, geometry)
    _seed_crystal(sim, geometry)

    free_energy, mass = _march(sim)
    _assert_sane(free_energy, mass, float(sim.State.phi.min()))
    _assert_relaxes(free_energy)


def test_logpfc_extv_runs():
    """Reduced log-PFC with an external potential: spectral Laplacian, ETD1.

    No monotonicity assertion - see this module's docstring.
    """
    pytest.importorskip("cupy", reason="GPU runtime only")
    from pfc_models.models.logpfc_extv import simulation as sim_mod

    geometry = sim_mod.BuildGeometry(Nx=NX, Ny=NY, Lx=LX, Ly=LY)
    params = sim_mod.BuildLogPFCModel_withVacancyForce_Parameters(
        Gamma=1.0, beta=0.5, q0=Q0, epsilon=-0.9, g=-2.0,
        dt=DT, meanDensity=MEAN_DENSITY_SOLID, V0=-1e-3, potWidth=10.0,
    )
    sim = sim_mod.BuildCupyGPU_Simulation(params, geometry)
    _seed_crystal(sim, geometry)

    free_energy, mass = _march(sim)
    _assert_sane(free_energy, mass, float(sim.State.phi.min()))


def test_geometry_contract():
    """The substrate geometry and kernels provide what the contract documents."""
    from pfc_models.discretization.spectral.geometry_2d import Build2DGeometry
    from pfc_models.discretization.spectral.kernels import (
        BuildDerivativeKernels, kernel_etd, kernel_local,
    )

    geometry = Build2DGeometry(Nx=64, Ny=64, Lx=32.0, Ly=32.0)
    for name in ("Nx", "Ny", "Lx", "Ly", "dx", "dy", "x", "y", "X", "Y",
                 "kx", "ky", "KX", "KY", "K2"):
        assert hasattr(geometry, name), f"geometry is missing {name}"
    assert np.allclose(geometry.K2, geometry.KX**2 + geometry.KY**2)

    kernels = BuildDerivativeKernels(geometry)
    assert np.allclose(kernels.d2_dlap, -(geometry.KX**2 + geometry.KY**2)), (
        "d2_dlap is not the spectral Laplacian"
    )
    assert np.isclose(kernel_local(geometry, w=2 * np.pi).max(), 1.0)

    linear, nonlinear = kernel_etd(kernels.d2_dlap, DT)
    assert np.isfinite(linear).all() and np.isfinite(nonlinear).all()
    assert np.isclose(nonlinear.flat[0], DT), "ETD k=0 coefficient is not dt"
