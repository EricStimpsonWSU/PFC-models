"""frozen_q (pfc-models v0.3.0): frozen-field f(q), the Get_f_ABC state hook, the edge
pass, and agreement with Find_q0Star at cheap points."""
import numpy as np
import pytest

from pfc_models.models.logpfc_cpu import phase_diagram as ad
from pfc_models.phase_diagram import frozen_q as fq
from pfc_models.phase_diagram import q0star as q0s

CELL = {'uNx': 14, 'uNy': 24, 'Mx': 2, 'My': 1}
SMALL = {'uNx': 7, 'uNy': 12, 'Mx': 2, 'My': 1}
H2L = {'mode': 'H2L', 'A1': 0.9, 'A2': 0.0, 'A3': 0.0}
RELAX = q0s.RelaxSettings(dt=0.01, dtMax=0.1, dtRate=1e-5, dtGrow=2.0, dtChecks=10, totalT=1e6,
                          stepT=100.0, divs=10, maxDepth=4, requiredFitness=0.99, minimumA=1e-6)
QS = q0s.QScanSettings(passes=3, Deltaq=0.02, q00_Steps=21, Deltaq_Refinement=0.05)


def _f_adapter(phi, params, structure, cell, temp, q):
    sim = ad.build(params, structure, cell, H2L, temp, float(phi.mean()), q, 0.01, state=np.array(phi))
    return float(ad.free_energy(sim))


@pytest.mark.parametrize('beta,g,temp', [(0.5, -2.0, -0.9), (0.25, 1.0, -3.0), (1.0, 0.0, -2.0)])
def test_frozen_f_equals_adapter(beta, g, temp):
    params = {'beta': beta, 'g': g, 'v0': 1.0}
    rng = np.random.default_rng(1)
    sim = ad.build(params, 'triangle', CELL, H2L, temp, 0.7, 1.0, 0.01)
    phi = ad.get_state(sim) * (1 + 0.05*rng.standard_normal(ad.get_state(sim).shape))
    ff = ad.frozen_field(phi, params, temp, CELL)
    for q in (0.7, 0.93, 1.0, 1.08, 1.3):
        fa = _f_adapter(phi, params, 'triangle', CELL, temp, q)
        assert abs(float(ff.f(q)) - fa) <= 1e-12*max(1.0, abs(fa))
    qs = np.linspace(0.05, 2.0, 39001)
    assert abs(qs[np.argmin(ff.f(qs))] - ff.q_opt()) < 1e-4
    clean = ad.frozen_field(ad.get_state(sim), params, temp, CELL)   # one-mode ansatz: q_opt = 1
    assert abs(clean.q_opt() - 1.0) < 1e-12


def test_get_f_abc_state_hook():
    params = {'beta': 0.5, 'g': -2.0, 'v0': 1.0}
    m = q0s.MinimizeFreeEnergy(ad, params, 'triangle', SMALL, H2L, RELAX, QS)
    f, fit, *_ = m.Get_f_ABC(-0.9, 0.62, 1.0)
    phi = m.last_state
    assert phi is not None and abs(phi.mean() - 0.62) < 1e-12
    assert abs(_f_adapter(phi, params, 'triangle', SMALL, -0.9, 1.0) - f) < 1e-14
    f2, *_ = m.Get_f_ABC(-0.9, 0.62, 1.0, state=phi)   # restart from a converged field
    assert abs(f2 - f) <= 1e-11 and m.last_steps < 5000


def test_edge_helpers():
    grid = [0.1, 0.2, 0.3, 0.4, 0.5]
    res = {0.1: {'status': 'poor_fit'}, 0.2: {'status': 'ok', 'q': 0.97, 'f': -1.0},
           0.3: {'status': 'ok', 'q': 0.98, 'f': -1.1}, 0.4: {'status': 'poor_fit'},
           0.5: {'status': 'poor_fit'}}
    assert fq.edge_walks(grid, res) == [(1, -1), (2, 1)]
    calls = []

    def solve(p, q):
        calls.append((p, q))
        return {'status': 'ok' if p == 0.4 else 'poor_fit', 'q': q, 'f': -2.0}
    recs = fq.edge_walk(solve, grid, res, 2, 1, maxPast=3)
    assert calls == [(0.4, 0.98), (0.5, 0.98)]
    merged = fq.merge_edge(res, recs)
    assert merged[0.4]['status'] == 'ok' and merged[0.4]['edge_from'] == 0.3
    assert merged[0.5]['status'] == 'poor_fit'


# cheap points (7x12-2x1 cell) where both methods accept the state. On this coarse cell
# stripe and honeycomb candidates are rejected (poor_fit) or fail Find_q0Star, so the
# cheap check is triangle-only; the production-cell comparison is in the paper repo.
AGREE = [(0.5, -2.0, 'triangle', -0.9, 0.62), (0.5, -2.0, 'triangle', -0.9, 0.66)]


@pytest.mark.parametrize('beta,g,structure,temp,phi', AGREE)
def test_agrees_with_find_q0star(beta, g, structure, temp, phi):
    params = {'beta': beta, 'g': g, 'v0': 1.0}
    q_f, f_f = q0s.MinimizeFreeEnergy(ad, params, structure, SMALL, H2L, RELAX, QS).Find_q0Star(temp, phi)
    m = fq.MinimizeFrozenQ(ad, params, structure, SMALL, H2L, RELAX)
    q, f = m.Find_q0Star(temp, phi)
    assert m.last_result['status'] == 'ok'
    assert abs(q - q_f) < 1e-5
    assert abs(f - f_f) < 1e-9
    assert abs(m.last_result['f_frozen'] - f) < 1e-10
