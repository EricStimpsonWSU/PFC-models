"""Regression test for logpfc_cpu: fixed seed, asserted free energy at fixed steps.

The point is not accuracy - it is detecting the day a change silently alters the
physics. The references were recorded from the ORIGINAL implementation
(warrior/phase-diagram/Simulation.py, BuildFFTW_CPU_Simulation, unmodified), which
P02-01 (warrior/phase-diagram/reconciliation.md sec 2.1) showed bit-identical to
this model through 1e4 steps.

Configuration (all of it load-bearing - change any line and the references are void):

    Nx, Ny = 28, 24 on Lx = 2 a, Ly = h   (a = 4 pi / (sqrt(3) q), h = 4 pi / q)
    q = 0.9648639799649456                (strain-free q0* of the triangle at phi = 0.6326)
    Gamma = 1, beta = 0.5, q0 = 1, epsilon = -0.9, g = -2, dt = 0.1, nThreads = 1
    phi = 0.6326 (1 + 0.15 (2 cos(sqrt(3)/2 q x) cos(q y / 2) + cos(q y)))
          + 1e-3 N(0, 1) from numpy.random.default_rng(7), mean restored to 0.6326

Step 10 is far from the relaxed state and is the sensitive check; by step 1000
the free energy is at its fixed point to about 1e-9.

Recorded 2026-11-09 with pyfftw 0.15.1 / numba 0.68.0 / NumPy 2.5.4, float64.
"""
import numpy as np
import pytest

pytest.importorskip("pyfftw", reason="CPU runtime only")
pytest.importorskip("numba", reason="CPU runtime only")

REFERENCE_FREE_ENERGY = {
    10: -0.5162767800170736,
    100: -0.554845641895177,
    1000: -0.5548456430520675,
}
REFERENCE_STEP = 1000
RTOL = 1e-10

Q = 0.9648639799649456
MEAN_DENSITY = 0.6326


def _build():
    from pfc_models.models.logpfc_cpu import simulation as sim_mod

    a, h = 4 * np.pi / (np.sqrt(3) * Q), 4 * np.pi / Q
    geometry = sim_mod.Build2DGeometry(Nx=28, Ny=24, Lx=2 * a, Ly=h)
    params = sim_mod.BuildLogPFCModelParameters(
        Gamma=1.0, beta=0.5, q0=1.0, epsilon=-0.9, g=-2.0, dt=0.1, meanDensity=MEAN_DENSITY,
    )
    sim = sim_mod.BuildCupyGPU_Simulation(params, geometry, 1)
    X, Y = geometry.X, geometry.Y
    phi = MEAN_DENSITY * (1 + 0.15 * (2 * np.cos(np.sqrt(3) / 2 * Q * X) * np.cos(Q * Y / 2)
                                      + np.cos(Q * Y)))
    phi += 1e-3 * np.random.default_rng(7).standard_normal(phi.shape)
    phi += MEAN_DENSITY - phi.mean()
    sim.State.phi[...] = phi
    return sim


def test_free_energy_at_fixed_step():
    sim = _build()
    done = 0
    for step in sorted(REFERENCE_FREE_ENERGY):
        for _ in range(step - done):
            sim.TimeStep()
        done = step
        assert sim.CalculateFreeEnergy() == pytest.approx(REFERENCE_FREE_ENERGY[step], rel=RTOL), step
    assert sim.phi.mean() == pytest.approx(MEAN_DENSITY, abs=1e-12)


if __name__ == "__main__":
    sim = _build()
    done = 0
    for step in sorted(REFERENCE_FREE_ENERGY):
        for _ in range(step - done):
            sim.TimeStep()
        done = step
        print(step, repr(sim.CalculateFreeEnergy()))
