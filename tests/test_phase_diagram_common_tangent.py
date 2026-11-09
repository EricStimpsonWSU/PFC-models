"""Tests for pfc_models.phase_diagram.common_tangent (model-agnostic coexistence).

No model adapter is imported: the log-PFC uniform (liquid) branch is defined
inline, f_l = phi ln phi + (beta + eps)/2 phi^2 + g/3 phi^3 + phi^4/4 with
mu_l = df_l/dphi (the +1 included), beta = 0.5, g = -2.

Reference (a): the eps = -0.9 triangle table stored with the original sandbox
(phase-diagram-refs.json, stored_tables.tables['-0.9'], the 30 finite points
embedded below) and the tangent of `_test_common_tangent_liquid-triangle.ipynb`
cells 9-20: phi_l = 0.2984542712985943, phi_s = 0.6326051770348774,
mu = -0.480085, omega = -0.251144. Measured 2026-11-09 with the analytic liquid:
phi_l + 2.064e-9, phi_s - 2.612e-9, mu - 3.6e-7, omega + 3.6e-7. The phi
differences are the notebook's spline representation of the liquid (with the
notebook's two liquid splines this module reproduces it to 2e-15, tested
below); the independent P02-01 fsolve values (0.2984542734, 0.6326051744,
q 0.9648640009) agree with the analytic-liquid tangent to 1e-10.

(b)-(f) use closed forms: parabolas f = k/2 (phi - c)^2 + e have
omega(mu) = -mu^2/(2k) - c mu + e, so every tangent solves a quadratic, and a
not-a-knot cubic spline reproduces them exactly.
"""

import csv
import io

import numpy as np
import pytest
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve

from pfc_models.phase_diagram.common_tangent import (
    STATUSES,
    analytic_branch,
    common_tangent,
    lower_envelope,
    maxwell_lines,
    tabulated_branch,
)

BETA, G = 0.5, -2.0

# stored_tables.tables['-0.9']: finite points only, as (phi, fmin, q0star)
TRIANGLE_EPS_M0P9 = [
    (0.52, -0.49781352633568954, 0.8272050314262395),
    (0.53, -0.5029052351329709, 0.8585318535406156),
    (0.54, -0.5080597030731432, 0.8814988472426624),
    (0.55, -0.5132387200239727, 0.8991144439215384),
    (0.56, -0.5184173253870454, 0.9131339402925371),
    (0.5700000000000001, -0.523578059452446, 0.9245888704685505),
    (0.5800000000000001, -0.5287079043888703, 0.9341195965605054),
    (0.5900000000000001, -0.5337966695577224, 0.9421457504851904),
    (0.6, -0.538836095408525, 0.9489552986944785),
    (0.61, -0.5438193276300246, 0.9547535373887293),
    (0.62, -0.5487405921806928, 0.9596914905660974),
    (0.63, -0.5535949848941154, 0.9638831589013295),
    (0.64, -0.5583783294201533, 0.967416429895785),
    (0.65, -0.5630870775482224, 0.9703602226924533),
    (0.66, -0.5677182367297171, 0.9727693072359255),
    (0.67, -0.5722693155906684, 0.9746876370737136),
    (0.68, -0.5767382816724322, 0.9761507015922278),
    (0.6900000000000001, -0.581123527698814, 0.9771872131536835),
    (0.7000000000000001, -0.5854238439404774, 0.9778203305990636),
    (0.7100000000000001, -0.5896383950571178, 0.9780685501940378),
    (0.72, -0.5937667003280741, 0.9779463512890046),
    (0.73, -0.5978086165395775, 0.9774646548553036),
    (0.74, -0.6017643230457951, 0.9766311333150237),
    (0.75, -0.6056343087012251, 0.9754503960592795),
    (0.76, -0.6094193604997266, 0.9739240646357278),
    (0.77, -0.6131205538684742, 0.9720507427963175),
    (0.78, -0.6167392446666585, 0.9698258785517508),
    (0.79, -0.6202770630407647, 0.9672415066594576),
    (0.8, -0.6237359094017879, 0.9642858498936953),
    (0.81, -0.6271179529294553, 0.9609427439179395),
]

REF_PHI_L = 0.2984542712985943
REF_PHI_S = 0.6326051770348774
REF_MU = -0.480085
REF_OMEGA = -0.251144
FSOLVE_P0201 = {"phi_l": 0.2984542734, "phi_s": 0.6326051744, "q": 0.9648640009}
NOTEBOOK_Q_AT_REF_PHI_S = 0.964864001830006  # notebook cell 20 output


def f_liquid(phi, eps):
    phi = np.asarray(phi, dtype=float)
    return phi * np.log(phi) + 0.5 * (BETA + eps) * phi**2 + G / 3.0 * phi**3 + 0.25 * phi**4


def mu_liquid(phi, eps):
    phi = np.asarray(phi, dtype=float)
    return 1.0 + np.log(phi) + (BETA + eps) * phi + G * phi**2 + phi**3


def d2f_liquid(phi, eps):
    phi = np.asarray(phi, dtype=float)
    return 1.0 / phi + (BETA + eps) + 2.0 * G * phi + 3.0 * phi**2


def liquid(eps, lo, hi, exact_d2f=False, name="liquid"):
    return analytic_branch(name, lambda p: f_liquid(p, eps), lambda p: mu_liquid(p, eps), lo, hi,
                           d2f=(lambda p: d2f_liquid(p, eps)) if exact_d2f else None)


def triangle(mirror=None):
    phi, f, q = (np.array(c) for c in zip(*TRIANGLE_EPS_M0P9))
    if mirror is not None:
        phi = mirror - phi
    return tabulated_branch("triangle", phi, f, q)


def parabola(name, k, c, e, lo, hi, tabulated=False, q0star=None, h=0.01):
    def f(p):
        return 0.5 * k * (np.asarray(p, dtype=float) - c) ** 2 + e

    if tabulated:
        x = np.linspace(lo, hi, int(round((hi - lo) / h)) + 1)
        return tabulated_branch(name, x, f(x), None if q0star is None else q0star(x))
    return analytic_branch(name, f, lambda p: k * (np.asarray(p, dtype=float) - c), lo, hi,
                           d2f=lambda p: np.full(np.shape(p), float(k)))


def parabola_tangents(a, b):
    """All exact tangents (phi_a, phi_b, mu, omega) between parabolas a=(k,c,e), b."""
    (ka, ca, ea), (kb, cb, eb) = a, b
    qa = -(0.5 / ka - 0.5 / kb)
    mus = [-(ea - eb) / -(ca - cb)] if qa == 0 else list(np.roots([qa, -(ca - cb), ea - eb]).real)
    return [(ca + m / ka, cb + m / kb, m, -m * m / (2 * ka) - m * ca + ea) for m in mus]


# ------------------------------------------------------------------------------ (a)


def test_stored_triangle_table_reports_concave_low_edge():
    tri = triangle()
    assert len(tri.segments) == 1 and tri.gaps == () and tri.dropped == ()
    assert len(tri.pieces) == 1
    (lo, hi), = tri.nonconvex
    assert lo == 0.52
    assert hi == pytest.approx(0.54942, abs=1e-5)
    assert tri.pieces[0].lo == hi and tri.pieces[0].hi == 0.81


def test_liquid_triangle_tangent_eps_m0p9_reproduces_reference():
    liq, tri = liquid(-0.9, 1e-3, 0.52), triangle()
    r = common_tangent(liq, tri)
    assert r.status == "ok" and r.supporting
    assert abs(r.phi_a - REF_PHI_L) <= 3e-9
    assert abs(r.phi_b - REF_PHI_S) <= 3e-9
    assert abs(r.mu - REF_MU) <= 1e-6
    assert abs(r.omega - REF_OMEGA) <= 1e-6
    assert r.residual < 1e-14
    assert abs(r.phi_a - FSOLVE_P0201["phi_l"]) < 1e-9
    assert abs(r.phi_b - FSOLVE_P0201["phi_s"]) < 1e-9
    assert r.q0star_a is None and abs(r.q0star_b - FSOLVE_P0201["q"]) < 1e-9
    # argument order does not matter
    s = common_tangent(tri, liq)
    assert (s.phi_a, s.phi_b, s.mu) == (r.phi_b, r.phi_a, r.mu)
    # the liquid over its whole range (concave band and high-density piece) changes nothing
    full = liquid(-0.9, 1e-3, 1.6)
    assert len(full.pieces) == 2 and len(full.nonconvex) == 1
    u = common_tangent(full, tri)
    assert (u.phi_a, u.phi_b) == (r.phi_a, r.phi_b)


def test_notebook_liquid_representation_reproduces_notebook_to_roundoff():
    # cells 9 and 11: CubicSpline of the analytic f and, separately, of the analytic mu
    x = np.linspace(0.01, 0.52, int((0.52 - 0.01) / 0.01), endpoint=False)
    f_spl, mu_spl = CubicSpline(x, f_liquid(x, -0.9)), CubicSpline(x, mu_liquid(x, -0.9))
    liq = analytic_branch("liquid", f_spl, mu_spl, x[0], x[-1])
    r = common_tangent(liq, triangle())
    assert r.status == "ok"
    assert abs(r.phi_a - REF_PHI_L) < 1e-13
    assert abs(r.phi_b - REF_PHI_S) < 1e-13
    assert abs(r.q0star_b - NOTEBOOK_Q_AT_REF_PHI_S) < 1e-13


def test_envelope_liquid_triangle_matches_pairwise_tangent():
    liq, tri = liquid(-0.9, 1e-3, 0.52), triangle()
    env = lower_envelope([liq, tri], temp=-0.9)
    assert [w.name for w in env.windows] == ["liquid", "triangle"]
    (c,) = env.coexistence
    r = common_tangent(liq, tri)
    assert c.stable and (c.a, c.b) == ("liquid", "triangle")
    assert (c.phi_a, c.phi_b, c.mu, c.omega) == (r.phi_a, r.phi_b, r.mu, r.omega)
    assert env.metastable == () and env.warnings == ()


# ------------------------------------------------------------------------------ (b)


STRIPE = (4.0, 0.85, -0.66)
HONEYCOMB = (6.0, 1.20, -0.75)


@pytest.mark.parametrize("tabulated", [False, True])
def test_parabola_solid_solid_tangent_is_exact(tabulated):
    a = parabola("stripe", *STRIPE, 0.65, 1.05, tabulated, q0star=lambda x: 0.98 - 0.1 * (x - 0.85))
    b = parabola("honeycomb", *HONEYCOMB, 0.95, 1.45, tabulated)
    (pa, pb, mu, om), = [t for t in parabola_tangents(STRIPE, HONEYCOMB) if 0.65 < t[0] < 1.05]
    r = common_tangent(a, b)
    assert r.status == "ok" and len(r.candidates) == 1
    tol = 1e-14
    assert abs(r.phi_a - pa) < tol and abs(r.phi_b - pb) < tol
    assert abs(r.mu - mu) < tol and abs(r.omega - om) < tol
    if tabulated:
        assert abs(r.q0star_a - (0.98 - 0.1 * (pa - 0.85))) < 1e-14
        assert r.q0star_b is None


@pytest.mark.parametrize("swap", [False, True])
def test_parabola_liquid_at_higher_density_is_exact(swap):
    hc, lq = (6.0, 1.10, -0.70), (2.0, 1.60, -0.72)
    a, b = parabola("honeycomb", *hc, 0.90, 1.35), parabola("liquid", *lq, 1.0, 2.2)
    (ps, pl, mu, om), = [t for t in parabola_tangents(hc, lq) if 0.9 < t[0] < 1.35]
    assert pl > ps
    r = common_tangent(b, a) if swap else common_tangent(a, b)
    got_s, got_l = (r.phi_b, r.phi_a) if swap else (r.phi_a, r.phi_b)
    assert r.status == "ok"
    assert abs(got_s - ps) < 1e-14 and abs(got_l - pl) < 1e-14
    assert abs(r.mu - mu) < 1e-14 and abs(r.omega - om) < 1e-14


# ------------------------------------------------------------------------------ (c)


def test_mirrored_parabolas_give_mirrored_tangent():
    c = 2.0
    a, b = parabola("stripe", *STRIPE, 0.65, 1.05), parabola("honeycomb", *HONEYCOMB, 0.95, 1.45)
    am = parabola("stripe", STRIPE[0], c - STRIPE[1], STRIPE[2], c - 1.05, c - 0.65)
    bm = parabola("honeycomb", HONEYCOMB[0], c - HONEYCOMB[1], HONEYCOMB[2], c - 1.45, c - 0.95)
    r, m = common_tangent(a, b), common_tangent(am, bm)
    assert m.status == "ok" and m.phi_a > m.phi_b
    assert abs(m.phi_a - (c - r.phi_a)) < 1e-14 and abs(m.phi_b - (c - r.phi_b)) < 1e-14
    assert abs(m.mu + r.mu) < 1e-14 and abs(m.omega - (r.omega + c * r.mu)) < 1e-14


def test_mirrored_liquid_triangle_puts_liquid_at_higher_density():
    c = 1.0
    r = common_tangent(liquid(-0.9, 1e-3, 0.52), triangle())
    liq_m = analytic_branch("liquid", lambda p: f_liquid(c - np.asarray(p), -0.9),
                            lambda p: -mu_liquid(c - np.asarray(p), -0.9), c - 0.52, c - 1e-3)
    m = common_tangent(liq_m, triangle(mirror=c))
    assert m.status == "ok" and m.phi_a > m.phi_b
    assert abs(m.phi_a - (c - r.phi_a)) < 1e-12 and abs(m.phi_b - (c - r.phi_b)) < 1e-12
    assert abs(m.mu + r.mu) < 1e-12 and abs(m.omega - (r.omega + c * r.mu)) < 1e-12
    assert abs(m.q0star_b - r.q0star_b) < 1e-12


# ------------------------------------------------------------------------------ (d)


def _cubic_concave_tail():
    # f'' = 1 - phi on [0, 3]: convex piece mu in [0, 0.5], concave part mu in [-1.5, 0.5]
    return analytic_branch("q", lambda p: np.asarray(p) ** 2 / 2 - np.asarray(p) ** 3 / 6,
                           lambda p: np.asarray(p) - np.asarray(p) ** 2 / 2, 0.0, 3.0)


@pytest.mark.parametrize(
    "case, status",
    [
        ("disjoint_mu", "no_overlap"),
        ("parallel", "no_root"),
        ("all_concave", "non_convex"),
        ("concave_overlap_only", "non_convex"),
        ("convex_self_pair", "no_root"),
    ],
)
def test_no_tangent_cases_return_status(case, status):
    if case == "disjoint_mu":
        a, b = parabola("a", 1.0, 0.0, 0.0, -0.1, 0.1), parabola("b", 1.0, 2.0, 0.0, 1.5, 1.7)
    elif case == "parallel":
        a, b = parabola("a", 1.0, 1.0, 0.0, 0.8, 1.2), parabola("b", 1.0, 1.0, 0.1, 0.7, 1.3)
    elif case == "all_concave":
        a = analytic_branch("c", lambda p: -np.asarray(p) ** 2, lambda p: -2 * np.asarray(p),
                            0.0, 1.0)
        b = parabola("b", 1.0, 1.0, 0.1, 0.7, 1.3)
        assert a.pieces == () and a.nonconvex == ((0.0, 1.0),)
    elif case == "concave_overlap_only":
        a, b = _cubic_concave_tail(), parabola("b", 1.0, 5.0, 0.0, 3.8, 4.5)
        assert abs(a.pieces[0].hi - 1.0) < 1e-10
    else:
        a = b = parabola("a", 1.0, 1.0, 0.0, 0.8, 1.2)
    r = common_tangent(a, b)
    assert status in STATUSES
    assert r.status == status and not r.ok
    assert np.isnan(r.phi_a) and np.isnan(r.mu) and r.message


# ------------------------------------------------------------------------------ (e)


PA, PB, PC = (4.0, 0.3, 0.0), (4.0, 0.6, -0.05), (4.0, 0.9, 0.0)
DOM = {"A": (0.0, 0.5), "B": (0.35, 0.85), "C": (0.7, 1.2)}


def three(eb=PB[2], kb=PB[0], dom_b=DOM["B"]):
    return [parabola("A", *PA, *DOM["A"]), parabola("B", kb, PB[1], eb, *dom_b),
            parabola("C", *PC, *DOM["C"])]


@pytest.mark.parametrize("grid", [None, np.linspace(0.0, 1.2, 601)])
def test_envelope_excludes_metastable_tangent(grid):
    env = lower_envelope(three(), grid)
    assert [w.name for w in env.windows] == ["A", "B", "C"]
    ab, bc = env.coexistence
    for got, ref in ((ab, parabola_tangents(PA, PB)[0]), (bc, parabola_tangents(PB, PC)[0])):
        assert got.stable and got.status == "ok" and not got.same_branch
        assert np.allclose((got.phi_a, got.phi_b, got.mu, got.omega), ref, atol=1e-14, rtol=0)
    assert env.windows[1].phi_lo == ab.phi_b and env.windows[1].phi_hi == bc.phi_a
    (ac,) = env.metastable
    assert (ac.a, ac.b, ac.stable, ac.below) == ("A", "C", False, "B")
    assert abs(ac.mu) < 1e-14 and abs(ac.phi_a - 0.3) < 1e-14 and abs(ac.phi_b - 0.9) < 1e-14
    assert env.warnings == ()


def test_envelope_refinement_inserts_missed_and_removes_spurious_windows():
    # narrow deep B between grid points: the hull misses it, the exact check inserts it
    coarse = lower_envelope(three(eb=-0.001, kb=400.0), np.linspace(0.0, 1.2, 13))
    fine = lower_envelope(three(eb=-0.001, kb=400.0))
    assert [w.name for w in coarse.windows] == ["A", "B", "C"]
    for c, f in zip(coarse.coexistence, fine.coexistence, strict=True):
        assert (c.phi_a, c.phi_b, c.mu) == (f.phi_a, f.phi_b, f.mu)
    # B above the A-C tangent but below the coarse chord: its window has crossing bridges
    env = lower_envelope(three(eb=0.0005, kb=400.0, dom_b=(0.55, 0.65)),
                         [0.0, 0.25, 0.35, 0.6, 0.85, 0.95, 1.2])
    assert [w.name for w in env.windows] == ["A", "C"]
    (ac,) = env.coexistence
    assert abs(ac.mu) < 1e-14 and abs(ac.phi_a - 0.3) < 1e-14
    assert {(m.a, m.b) for m in env.metastable} == {("A", "B"), ("B", "C")}


def test_concave_liquid_gives_liquid_liquid_bridge():
    eps = -0.9
    spin = np.sort(np.roots([3.0, 2.0 * G, BETA + eps, 1.0]).real)[1:]  # positive roots
    exact = liquid(eps, 1e-3, 1.6, exact_d2f=True)
    numeric = liquid(eps, 1e-3, 1.6)
    assert np.allclose([exact.pieces[0].hi, exact.pieces[1].lo], spin, atol=1e-14, rtol=0)
    assert np.allclose([numeric.pieces[0].hi, numeric.pieces[1].lo], spin, atol=1e-9, rtol=0)
    env = lower_envelope([exact])
    (ll,) = env.coexistence
    assert ll.same_branch and not ll.across_gap and ll.stable and ll.supporting
    assert ll.phi_a < spin[0] < spin[1] < ll.phi_b
    assert ll.residual < 1e-14
    ref = fsolve(lambda v: [mu_liquid(v[0], eps) - mu_liquid(v[1], eps),
                            f_liquid(v[1], eps) - f_liquid(v[0], eps)
                            - mu_liquid(v[0], eps) * (v[1] - v[0])], [0.35, 1.45], xtol=1e-13)
    assert abs(ll.phi_a - ref[0]) < 1e-10 and abs(ll.phi_b - ref[1]) < 1e-10
    # with the triangle present the liquid-liquid tangent is metastable (triangle below)
    env2 = lower_envelope([numeric, triangle()])
    (meta,) = [m for m in env2.metastable if m.same_branch]
    assert meta.below == "triangle" and abs(meta.phi_a - ll.phi_a) < 1e-9


def test_nan_gaps_split_segments_and_flag_bridges_across_them():
    x = np.round(np.linspace(-1.5, 1.5, 61), 12)
    f = (x**2 - 1) ** 2
    f[np.abs(x) < 0.4] = np.nan
    f[2] = np.nan  # leaves a 2-point run at the low end, dropped
    dw = tabulated_branch("dw", x, f)
    assert [(s.lo, s.hi) for s in dw.segments] == [(-1.35, -0.4), (0.4, 1.5)]
    assert dw.gaps == ((-0.4, 0.4),) and dw.dropped == (-1.5, -1.45)
    assert [p.segment for p in dw.pieces] == [0, 1] and len(dw.nonconvex) == 2
    r = common_tangent(dw, dw)
    assert r.status == "ok" and r.same_branch and r.across_gap
    assert abs(r.phi_a + 1) < 1e-7 and abs(r.phi_b - 1) < 1e-7 and abs(r.mu) < 1e-12
    env = lower_envelope([dw])
    assert env.coexistence[0].across_gap and any("NaN gap" in w for w in env.warnings)


# ------------------------------------------------------------------------------ (f)


def test_maxwell_lines_brackets_triple_point():
    # e_B(T) = -0.05 + 0.11 T: B's window closes exactly at T* = 0.05 / 0.11
    temps = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75]
    envs = {t: lower_envelope(three(eb=-0.05 + 0.11 * t), temp=t) for t in temps}
    ml = maxwell_lines(envs)
    assert ml.temps == tuple(temps)
    pairs = [(r["temp"], r["a"], r["b"]) for r in ml.rows]
    assert pairs == [p for t in temps for p in
                     ([(t, "A", "B"), (t, "B", "C")] if t < 0.05 / 0.11 else [(t, "A", "C")])]
    (tp,) = ml.triple_points
    assert tp.phases == ("A", "B", "C")
    assert (tp.temp_lo, tp.temp_hi, tp.temp_open, tp.temp_closed) == (0.45, 0.6, 0.45, 0.6)
    assert tp.width_open > 0 > tp.width_closed
    assert abs(tp.temp_estimate - 0.05 / 0.11) < 1e-12
    # iterable of envelopes with temp set gives the same table
    assert maxwell_lines(list(envs.values())).triple_points == ml.triple_points
    buf = io.StringIO()
    ml.write_csv(buf)
    got = list(csv.DictReader(io.StringIO(buf.getvalue())))
    assert len(got) == len(ml.rows) and float(got[0]["phi_b"]) == ml.rows[0]["phi_b"]
