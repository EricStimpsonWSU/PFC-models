"""Tests for pfc_models.phase_diagram.q0star and the logpfc_cpu / stdpfc_cpu adapters.

(a) contract: a toy adapter whose state is a TWO-field tuple and whose relaxed
    f(q00) has a known minimum drives MinimizeFreeEnergy, Find_q0Star and
    scan_density (status codes, checkpoint, resume) through the adapter names alone;
(b) logpfc_cpu Get_f_ABC reproduces the ORIGINAL code at one point per structure;
(c) logpfc_cpu Find_q0Star reproduces q0* and fmin, and an exception point maps
    to 'no_local_minimum' through scan_density;
(d) logpfc_cpu analytics: critical point, the legacy criticalTemp column, the
    spinodal and one-mode saddle-node bounds, the uniform free energy;
(e) stdpfc_cpu: at g = 0 the triangle at -nbar is the mirror of the honeycomb at
    +nbar; closed-form analytics.

References for (b) and (c) were captured from the unmodified
warrior/phase-diagram code (phase-diagram-refs.json, 2026-11-09, pyfftw 0.15.1,
numba 0.68.0, NumPy 2.5.4, 1 thread) with MinimizeFreeEnergy(8, 1, 0, lat, 0, 0, 2):
model_vals[8], geometry_vals[1] x system_geometry_vals[0], mode_vals[0],
run_vals[2], fit_vals[0]. The relocation is bit-identical, so they are asserted
with ==. The (d) bounds are copied from warrior/phase-diagram/phase_stability_bounds.csv
(hand-off addendum, 31 Oct 2026); that folder is temporary, so the values live here.

Cost: (b) about 6 s, (c) about 40 s on an idle laptop core.
"""
import inspect
import os
import pathlib
import shutil
import tempfile
import uuid

import numpy as np
import pytest

from pfc_models.phase_diagram import q0star as Q

# ---------------------------------------------------------------- shared settings

RELAX = Q.RelaxSettings(dt=0.1, totalT=1e6, stepT=1e2, divs=10, maxDepth=4,
                        requiredFitness=0.99, minimumA=1e-6)            # legacy run_vals[2]
QSCAN = Q.QScanSettings(passes=3, Deltaq=0.02, q00_Steps=21, Deltaq_Refinement=0.05)  # fit_vals[0]
CELL = dict(uNx=14, uNy=24, Mx=2, My=1)                                 # geometry_vals[1] x system_geometry_vals[0]
ANSATZ = dict(mode='L2H', A1=1e-5, A2=1e-6, A3=1e-7)                     # mode_vals[0]
LOG_PARAMS = dict(beta=0.5, g=-2.0, v0=1.0)                              # model_vals[8]

GET_F_ABC_REFS = [  # structure, eps, phi, q00, f, fitness, A, B, C, timesteps
    ('triangle', -0.9, 0.63, 0.9638831589013295, -0.553594984894113, 0.9997633941916881, 0.33320928950407136, 0.03198014374195086, 0.003245425514830756, 2112),
    ('honeycomb', -0.9, 1.2, 1.0, -0.7518361494024, 0.9999247733487598, 0.14468529726021068, 0.007786537467545442, 0.0012330653960552866, 1356),
    ('stripe', -0.9, 0.88, 1.0, -0.6617397486488582, 0.99983068778873, 0.4718018020045463, 0.021804312871064338, 0.005226927920294405, 2111),
]
FIND_Q0STAR_REF = ('triangle', -0.9, 0.63, 0.9638831589507376, -0.5535949848941145)  # eps, phi, q0*, fmin
EXCEPTION_REF = ('triangle', -0.9, 0.52, 'No local minimum found for q00.')
CRITICAL_TEMP_TABLE = [  # legacy model_vals: beta, g, criticalTemp
    (1.0, -3, 2.064165490400111),
    (1.0, -2, 0.0603891377557797),
    (1.0, -1, -1.4975406761582464),
    (1.0, 0, -2.7256891399289986),
    (1.0, 1, -3.7290200842057306),
    (1.0, 2, -4.579735943195909),
    (1.0, 3, -5.324661558934498),
    (0.5, -3, 2.064165490400111),
    (0.5, -2, 0.0603891377557797),
    (0.5, -1, -1.4975406761582464),
    (0.5, 0, -2.7256891399289986),
    (0.5, 1, -3.7290200842057306),
    (0.5, 2, -4.579735943195909),
    (0.5, 3, -5.324661558934498),
]
STABILITY_BOUNDS_CSV = [  # phase_stability_bounds.csv: eps, phi_minus, phi_plus, landau_tri_low, landau_hc_high
    (0.05, 0.8333333333333324, 0.9300735254367731, 0.8278838706570207, 0.9360125588507695),
    (0.0, 0.7675918792439986, 0.9999999999999978, 0.7554992850198972, 1.014929548555121),
    (-0.2, 0.6524325446875517, 1.1321668662225268, 0.631887250872099, 1.1649090296597184),
    (-0.4, 0.5838532526597833, 1.218155604616871, 0.5606585549669681, 1.2626859101321015),
    (-0.6, 0.5321777251171093, 1.2876064091651318, 0.5082496704574803, 1.3416502300819044),
    (-0.8, 0.49021375435214665, 1.3476743272029577, 0.46645677816574976, 1.409907314718121),
    (-0.9, 0.47182259148403954, 1.3752289917267846, 0.4483417470214427, 1.44120201692306),
    (-1.2, 0.4242592654794263, 1.450673074588675, 0.40200033046424577, 1.5268249929704056),
    (-1.5, 0.3852464105732987, 1.5180570087930465, 0.3644592480446592, 1.6032198262946251),
]


# ---------------------------------------------------------------- (a) toy adapter

class _ToySim:
    """Two fields (u, v) decaying exponentially towards a q00-dependent floor."""
    def __init__(self, f_floor, fitness, amplitude, dt, u, v):
        self.f_floor, self.fitness, self.amplitude, self.dt = f_floor, fitness, amplitude, dt
        self.u, self.v, self.t = u, v, 0.0


class ToyAdapter:
    """Implements exactly the names q0star calls; nothing else.

    temp shifts the floor; meanDensity selects the regime:
      0 < rho < 1   f_floor = temp + rho^2 + K (q00 - QSTAR)^2, fitness 1 - (q00 - QSTAR)^2
      rho == 2      f_floor linear in q00 (no local minimum)
      rho == 3      fitness 0.5 everywhere (insufficient points)
      rho < 0       build raises ValueError (error)
    """
    SLUG = 'toy'
    STRUCTURES = {'blob': 'blob'}
    QSTAR = 1.03
    K = 1.0

    def __init__(self):
        self.builds = 0
        self.builds_from_state = 0
        self.steps = 0
        self.interrupt_at = None

    def validate_params(self, params):
        return dict(params)

    def validate_cell(self, cell):
        return dict(cell)

    def validate_ansatz(self, ansatz):
        return dict(ansatz)

    def build(self, params, structure, cell, ansatz, temp, meanDensity, q00, dt, state=None, nThreads=1):
        if self.interrupt_at is not None and meanDensity == self.interrupt_at:
            raise KeyboardInterrupt
        if meanDensity < 0:
            raise ValueError('negative density')
        self.builds += 1
        if meanDensity == 2.0:
            floor, fitness = temp + q00, 1.0
        elif meanDensity == 3.0:
            floor, fitness = temp + (q00 - self.QSTAR)**2, 0.5
        else:
            floor = temp + meanDensity**2 + self.K*(q00 - self.QSTAR)**2
            fitness = 1.0 - (q00 - self.QSTAR)**2
        if state is None:
            u, v = np.full(3, 0.5), np.full(5, -0.25)
        else:
            assert isinstance(state, tuple) and len(state) == 2
            self.builds_from_state += 1
            u, v = state[0].copy(), state[1].copy()
        return _ToySim(floor, fitness, 0.3, dt, u, v)

    def get_state(self, sim):
        return (sim.u.copy(), sim.v.copy())

    def set_state(self, sim, state):
        sim.u[...] = state[0]
        sim.v[...] = state[1]

    def get_time(self, sim):
        return sim.t

    def set_time(self, sim, t):
        sim.t = t

    def get_dt(self, sim):
        return sim.dt

    def step(self, sim):
        self.steps += 1
        sim.u *= 1 - 0.5*sim.dt
        sim.v *= 1 - 1.0*sim.dt
        sim.t += sim.dt

    def free_energy(self, sim):
        return sim.f_floor + np.mean(sim.u**2) + np.mean(sim.v**2)

    def diagnose(self, sim, structure, cell):
        return Q.Diagnostics(fitness=sim.fitness, amplitudes=(sim.amplitude, 0.0), details={})


TOY_RELAX = Q.RelaxSettings(dt=0.1, totalT=1e4, stepT=10.0, divs=10, maxDepth=4,
                            requiredFitness=0.99, minimumA=1e-6)


def _toy_minimizer(adapter):
    return Q.MinimizeFreeEnergy(adapter, {}, 'blob', {}, {}, TOY_RELAX, QSCAN)


def test_q0star_knows_no_model():
    src = inspect.getsource(Q)
    for token in ('sim.Model', '.State', 'sim.phi', 'sim.n', 'phi_hat', 'kvector', 'from pfc_models.models',
                  'import pfc_models.models'):
        assert token not in src, token


def test_toy_get_f_abc_relaxes_two_field_state():
    ad = ToyAdapter()
    m = _toy_minimizer(ad)
    f, fitness, A, B = m.Get_f_ABC(temp=0.1, meanDensity=0.5, q00=1.0)
    assert f == pytest.approx(0.1 + 0.25 + (1.0 - ToyAdapter.QSTAR)**2, abs=1e-14)
    assert (fitness, A, B) == (1.0 - (1.0 - ToyAdapter.QSTAR)**2, 0.3, 0.0)
    # the rollback path ran: the stagnating block was rebuilt from the opaque tuple state
    assert ad.builds_from_state >= TOY_RELAX.maxDepth


def test_toy_find_q0star_finds_known_minimum_and_records():
    ad = ToyAdapter()
    m = _toy_minimizer(ad)
    recs = []
    q, fmin = m.Find_q0Star(temp=0.1, meanDensity=0.5, record=recs.append)
    assert q == pytest.approx(ToyAdapter.QSTAR, abs=1e-9)
    assert float(fmin) == pytest.approx(0.1 + 0.25, abs=1e-12)
    assert len(recs) == QSCAN.passes*QSCAN.q00_Steps
    assert sorted(recs[0]) == ['amplitudes', 'f', 'fitness', 'pass', 'q00']
    assert [r['pass'] for r in recs[::QSCAN.q00_Steps]] == list(range(QSCAN.passes))
    assert recs[0]['amplitudes'] == (0.3, 0.0)
    # record has no effect on the result
    q2, fmin2 = _toy_minimizer(ToyAdapter()).Find_q0Star(temp=0.1, meanDensity=0.5)
    assert (q2, float(fmin2)) == (q, float(fmin))


def test_toy_scan_density_status_codes():
    r = Q.scan_density(_toy_minimizer(ToyAdapter()), 0.1, [0.5, 2.0, 3.0, -1.0])
    assert list(r['status']) == ['ok', 'no_local_minimum', 'insufficient_points', 'error']
    assert list(r['message']) == ['', Q.MSG_NO_LOCAL_MINIMUM, Q.MSG_INSUFFICIENT_POINTS,
                                  'ValueError: negative density']
    assert r['q0star'][0] == pytest.approx(ToyAdapter.QSTAR, abs=1e-9)
    assert np.isnan(r['q0star'][1:]).all() and np.isnan(r['fmin'][1:]).all()
    assert np.isfinite(r['wall_s']).all()
    qs = r['qscan']
    # ok: 3 passes; no local minimum and insufficient points: 1 pass; error: none
    assert np.bincount(qs['phi_index'], minlength=4).tolist() == [63, 21, 21, 0]
    assert qs['amplitudes'].shape == (105, 2)
    assert set(qs) == {'phi_index', 'pass_index', 'q00', 'f', 'fitness', 'amplitudes'}


@pytest.fixture
def ckdir():
    """A fresh directory with default permissions. Not `tmp_path`: on Windows with
    Python >= 3.13, pytest's mode-0o700 directories get an owner-only ACL that
    sandboxed runners (AppContainer) cannot list."""
    path = os.path.join(tempfile.gettempdir(), f'q0star-ck-{uuid.uuid4().hex}')
    os.mkdir(path)
    yield pathlib.Path(path)
    shutil.rmtree(path, ignore_errors=True)


def test_toy_scan_density_checkpoint_and_resume(ckdir):
    tmp_path = ckdir
    densities = [0.5, 2.0, 0.25, 3.0]
    fresh = Q.scan_density(_toy_minimizer(ToyAdapter()), 0.1, densities)

    ad = ToyAdapter()
    ad.interrupt_at = 0.25
    with pytest.raises(KeyboardInterrupt):
        Q.scan_density(_toy_minimizer(ad), 0.1, densities, checkpoint=str(tmp_path))
    with np.load(tmp_path / Q.FMIN_FILE) as z:
        assert list(z['status']) == ['ok', 'no_local_minimum', '', '']
    assert sorted(os.listdir(tmp_path)) == [Q.FMIN_FILE, Q.QSCAN_FILE]

    ad = ToyAdapter()
    resumed = Q.scan_density(_toy_minimizer(ad), 0.1, densities, checkpoint=str(tmp_path))
    # only the two undone densities were computed: 3 passes + 1 pass of 21 evaluations
    assert ad.builds - ad.builds_from_state == 4*QSCAN.q00_Steps
    for k in ('phi', 'q0star', 'fmin', 'status', 'message'):
        np.testing.assert_array_equal(resumed[k], fresh[k])
    with np.load(tmp_path / Q.FMIN_FILE) as z:
        np.testing.assert_array_equal(z['status'], fresh['status'])
    with np.load(tmp_path / Q.QSCAN_FILE) as z:
        assert z['q00'].size == fresh['qscan']['q00'].size
        np.testing.assert_array_equal(np.sort(z['f']), np.sort(fresh['qscan']['f']))

    # a finished checkpoint is skipped entirely; a different density list is refused
    ad = ToyAdapter()
    Q.scan_density(_toy_minimizer(ad), 0.1, densities, checkpoint=str(tmp_path))
    assert ad.builds == 0
    with pytest.raises(ValueError):
        Q.scan_density(_toy_minimizer(ToyAdapter()), 0.1, [0.5], checkpoint=str(tmp_path))


# ---------------------------------------------------------------- (b), (c) logpfc_cpu vs the original

try:
    import numba  # noqa: F401
    import pyfftw  # noqa: F401
    HAVE_CPU = True
except ImportError:
    HAVE_CPU = False
cpu = pytest.mark.skipif(not HAVE_CPU, reason="CPU runtime only")

if HAVE_CPU:
    from pfc_models.models.logpfc_cpu import phase_diagram as LOG
    from pfc_models.models.logpfc_cpu import simulation as LOG_SIM
    from pfc_models.models.stdpfc_cpu import phase_diagram as STD


@pytest.fixture
def count_log_steps(monkeypatch):
    calls = {'n': 0}
    original = LOG_SIM.Simulation.TimeStep

    def counted(self):
        calls['n'] += 1
        original(self)
    monkeypatch.setattr(LOG_SIM.Simulation, 'TimeStep', counted)
    return calls


def _log_minimizer(structure):
    return Q.MinimizeFreeEnergy(LOG, LOG_PARAMS, structure, CELL, ANSATZ, RELAX, QSCAN)


@cpu
@pytest.mark.parametrize('ref', GET_F_ABC_REFS, ids=[r[0] for r in GET_F_ABC_REFS])
def test_logpfc_get_f_abc_matches_original(ref, count_log_steps):
    structure, eps, phi, q00, *expected, timesteps = ref
    got = _log_minimizer(structure).Get_f_ABC(temp=eps, meanDensity=phi, q00=q00)
    assert [float(x) for x in got] == expected
    assert count_log_steps['n'] == timesteps


@cpu
def test_logpfc_find_q0star_matches_original():
    structure, eps, phi, q0s, fmin = FIND_Q0STAR_REF
    q, f = _log_minimizer(structure).Find_q0Star(temp=eps, meanDensity=phi)
    assert (float(q), float(f)) == (q0s, fmin)


@cpu
def test_logpfc_exception_point_maps_to_status():
    structure, eps, phi, message = EXCEPTION_REF
    r = Q.scan_density(_log_minimizer(structure), eps, [phi])
    assert (str(r['status'][0]), str(r['message'][0])) == (Q.STATUS_NO_LOCAL_MINIMUM, message)
    assert np.isnan(r['q0star'][0]) and np.isnan(r['fmin'][0])


# ---------------------------------------------------------------- (d) logpfc_cpu analytics

@cpu
def test_logpfc_validate_params():
    assert LOG.validate_params(dict(beta=0.5, g=-2)) == dict(beta=0.5, g=-2.0, v0=1.0, Gamma=1.0, q0=1.0)
    for bad in (dict(beta=0.5, g=-2, v0=0.9), dict(beta=0.5), dict(beta=0.5, g=-2, gamma=1.0)):
        with pytest.raises(ValueError):
            LOG.validate_params(bad)


@cpu
def test_logpfc_critical_point():
    phi_c, eps_c = LOG.critical_point(dict(beta=0.5, g=-2.0))
    assert phi_c == pytest.approx(0.881268, abs=1e-6)
    assert eps_c == pytest.approx(0.060444, abs=1e-6)
    assert 6*phi_c**3 - 4*phi_c**2 - 1 == pytest.approx(0, abs=1e-14)


@cpu
def test_logpfc_critical_temp_column():
    # Measured 2026-11-09: the table sits BELOW the exact eps_c at every g, by
    # 8.3e-6 (g=0) to 3.29e-4 (g=+1); 1.79e-4 at g=-3, 5.4e-5 at g=-2. One-sided,
    # as a sampled maximum would be. The exact value reproduces the addendum and the
    # CSV below to 1e-12, so the table is the less accurate of the two.
    for beta, g, tabulated in CRITICAL_TEMP_TABLE:
        _, eps_c = LOG.critical_point(dict(beta=beta, g=g))
        assert 0 < eps_c - tabulated < 4e-4, (beta, g)


@cpu
def test_logpfc_spinodal_and_existence_bounds_match_addendum():
    # measured agreement: spinodal 2.9e-15, saddle-node roots 4.1e-13
    for eps, phi_minus, phi_plus, tri_low, hc_high in STABILITY_BOUNDS_CSV:
        lo, hi = LOG.spinodal(LOG_PARAMS, eps)
        assert (lo, hi) == pytest.approx((phi_minus, phi_plus), abs=1e-11)
        b = LOG.existence_bounds(LOG_PARAMS, eps)
        assert b['triangle'] == pytest.approx((tri_low, phi_plus), abs=1e-11)
        assert b['honeycomb'] == pytest.approx((phi_minus, hc_high), abs=1e-11)
        assert b['stripe'] == (lo, hi)
        w = LOG.scan_window('triangle', LOG_PARAMS, eps)
        assert w[0] < tri_low and w[1] > phi_plus


@cpu
def test_logpfc_bands_vanish_at_critical_point():
    phi_c, eps_c = LOG.critical_point(LOG_PARAMS)
    assert LOG.spinodal(LOG_PARAMS, eps_c + 1e-9) is None
    assert LOG.existence_bounds(LOG_PARAMS, eps_c + 1e-9) is None
    lo, hi = LOG.spinodal(LOG_PARAMS, eps_c - 1e-6)
    assert lo < phi_c < hi
    # k = 0: same phi_c, eps shifted by beta q0^4 (addendum: -0.44 for beta = 0.5)
    assert LOG.liquid_spinodal(LOG_PARAMS, eps_c - 0.5 + 1e-9) is None
    lo, hi = LOG.liquid_spinodal(LOG_PARAMS, eps_c - 0.5 - 1e-6)
    assert lo < phi_c < hi


@cpu
@pytest.mark.parametrize('slug, density', [('logpfc_cpu', 0.7), ('stdpfc_cpu', -0.3)])
def test_liquid_free_energy_is_the_models_uniform_state(slug, density):
    adapter = {'logpfc_cpu': LOG, 'stdpfc_cpu': STD}[slug]
    params, eps = dict(beta=0.5, g=-2.0), -0.9
    sim = adapter.build(params, 'triangle', CELL, ANSATZ, eps, density, 1.0, 0.1,
                        state=np.full((24, 28), density))
    assert float(adapter.free_energy(sim)) == pytest.approx(adapter.liquid_free_energy(density, params, eps), rel=1e-14)
    h = 1e-6
    numeric = (adapter.liquid_free_energy(density + h, params, eps)
               - adapter.liquid_free_energy(density - h, params, eps))/(2*h)
    assert adapter.liquid_chemical_potential(density, params, eps) == pytest.approx(numeric, rel=1e-8)


# ---------------------------------------------------------------- (e) stdpfc_cpu

STD_PARAMS = dict(beta=1.0, g=0.0)
STD_EPS, STD_NBAR = -0.25, 0.25   # inside the one-mode hexagonal window |nbar| <= sqrt(-5 eps/12) = 0.323


@cpu
def test_stdpfc_g0_triangle_mirrors_honeycomb():
    # Measured 2026-11-09: bit-identical tuple (f 0.020967377878349205, fitness
    # 0.9999995344356641, A 0.1324548908767042), 4111 steps each. Exact because
    # f[-n] = f[n] at g = 0, the dynamics are odd in n, and both structures
    # share one amplitude scale independent of the sign of nbar.
    tri = Q.MinimizeFreeEnergy(STD, STD_PARAMS, 'triangle', CELL, ANSATZ, RELAX, QSCAN)
    hc = Q.MinimizeFreeEnergy(STD, STD_PARAMS, 'honeycomb', CELL, ANSATZ, RELAX, QSCAN)
    a = tri.Get_f_ABC(temp=STD_EPS, meanDensity=-STD_NBAR, q00=1.0)
    b = hc.Get_f_ABC(temp=STD_EPS, meanDensity=+STD_NBAR, q00=1.0)
    assert [float(x) for x in a] == [float(x) for x in b]
    assert a[1] > 0.999 and a[2] > 0.1
    assert a[0] < STD.liquid_free_energy(STD_NBAR, STD_PARAMS, STD_EPS)


@cpu
def test_stdpfc_analytics_closed_form():
    assert STD.critical_point(dict(beta=1.0, g=-1.5)) == (0.5, 0.75)
    lo, hi = STD.spinodal(STD_PARAMS, STD_EPS)
    assert (lo, hi) == pytest.approx((-np.sqrt(-STD_EPS/3), np.sqrt(-STD_EPS/3)), abs=1e-15)
    b = STD.existence_bounds(STD_PARAMS, STD_EPS)
    assert b['triangle'][0] == pytest.approx(-np.sqrt(-5*STD_EPS/12), abs=1e-15)
    assert b['honeycomb'][1] == pytest.approx(np.sqrt(-5*STD_EPS/12), abs=1e-15)
    p, eps = dict(beta=1.0, g=0.8), -0.4
    lo, hi = STD.spinodal(p, eps)
    for x in (lo, hi):
        assert eps + 2*0.8*x + 3*x**2 == pytest.approx(0, abs=1e-14)
    assert STD.spinodal(p, 0.8**2/3 + 1e-9) is None
    lo, hi = STD.liquid_spinodal(p, eps - 1.0)
    for x in (lo, hi):
        assert 1.0 + eps - 1.0 + 2*0.8*x + 3*x**2 == pytest.approx(0, abs=1e-14)
    # no positivity clamp: the triangle window extends to negative densities
    lo_t, hi_t = STD.existence_bounds(STD_PARAMS, STD_EPS)['triangle']
    assert STD.scan_window('triangle', STD_PARAMS, STD_EPS) == (lo_t - 0.02, hi_t + 0.02)


@cpu
def test_stdpfc_validate_params():
    with pytest.raises(ValueError):
        STD.validate_params(dict(beta=1.0, g=0.0, v0=2.0))
