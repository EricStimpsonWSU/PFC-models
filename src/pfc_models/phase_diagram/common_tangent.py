"""Coexistence between free-energy branches of one conserved density.

Model-agnostic: nothing here knows which model produced a free-energy curve. A
*branch* is the free energy per unit area f(phi) of one phase as a function of
the mean density phi, either analytic (the uniform state of an adapter,
`analytic_branch`) or tabulated (the fmin(phi) points of a q0* scan,
`tabulated_branch`). Coexistence is the common tangent (Maxwell construction):
two densities phi_a, phi_b with equal chemical potential mu = f'(phi) and equal
grand potential omega = f - mu phi.

Scope: one conserved density. Models with several conserved densities (two
species, a density and a concentration) need a tangent-plane construction over
the vector of chemical potentials, which belongs in a sibling module, not a flag
here.

Method
------
A branch is split into *pieces*: contiguous density intervals on which f'' > 0,
so mu(phi) is strictly increasing and phi(mu) is single valued. Concave
intervals (f'' < 0, the branch's spinodal region) are detected and reported in
`Branch.nonconvex`; inversions phi(mu) are never taken across them. For a pair of
convex pieces, the grand potentials omega_a(mu) and omega_b(mu) are evaluated on
the overlap of the pieces' mu ranges; since d omega / d mu = -phi, their
difference is monotonic wherever the two pieces do not cross in phi, and each
sign change is refined with `scipy.optimize.brentq` in mu, inverting mu(phi) on
each piece with brentq as well. Nothing assumes which branch sits at the lower
density, so liquid-solid, solid-solid and solid-liquid with the liquid at the
higher density (honeycomb-liquid) are the same computation.

This replaces the notebook procedure of `_test_common_tangent_liquid-triangle.ipynb`
(cells 9-20 of the warrior phase-diagram sandbox): there the liquid was itself a
`CubicSpline` of the analytic f and a second, separate `CubicSpline` of the
analytic mu on a 0.01 grid, phi(mu) came from `PPoly.solve`, and the tangent was
the root of a further spline of the f difference over 1000 mu samples. Measured
at eps = -0.9 on the stored triangle table (beta = 0.5, g = -2): this module with
the notebook's two liquid splines reproduces the notebook's phi_l, phi_s to
3e-15, so the 1000-sample spline step was exact to round-off. With the analytic
liquid it gives phi_l = 0.2984542733625, phi_s = 0.6326051744225 (2.1e-9 and
-2.6e-9 from the notebook): that difference is the notebook's spline
representation of the liquid (f error -6.3e-10, mu error 6.1e-9 at phi_l), not
the solid. A liquid represented by a spline of f and *its* derivative is far
worse on a 0.01 grid (phi_l off by 3e-7): pass the analytic liquid as an
analytic branch whenever the adapter provides it.

Spline end condition. `tabulated_branch` uses scipy's default not-a-knot end
condition, as the notebook did. It matters at the 1e-9 level: on the eps = -0.9
table a natural spline moves phi_s by -5e-10 and a clamped (zero-slope) spline
by 8e-8 against not-a-knot. A not-a-knot cubic spline reproduces cubic data
exactly, which the tests use for closed-form checks.

Status codes
------------
`common_tangent` never raises for a pair without a tangent; it returns a
`TangentResult` whose `status` is one of

* ``ok`` - a tangent between two convex pieces was found.
* ``no_overlap`` - the full mu ranges of the two branches (concave parts
  included) do not overlap.
* ``no_root`` - convex pieces overlap in mu but omega_a - omega_b has no sign
  change there (one branch lies below the other over the whole overlap; also
  the result of a self-pair on a branch that is convex throughout).
* ``non_convex`` - a branch has no convex piece at all, or the mu ranges overlap
  only through concave parts, so no tangent between stable parts exists.

Envelope and Maxwell lines
--------------------------
`lower_envelope` takes every branch at one temperature, computes the lower
convex hull of their sampled points, reads off the single-phase windows and the
hull bridges in density order, refines each bridge with `common_tangent`, and
then checks the refined bridges exactly: a window whose two bridges cross is
removed, and a branch that dips below a bridge's tangent line is inserted as a
window. Every other pairwise tangent is reported as metastable, with the branch
that lies below its line. A bridge between two pieces of the same branch (a
liquid-liquid gap where the uniform branch is concave) is flagged
`same_branch`; one across a NaN gap of a tabulated branch is flagged
`across_gap` and warned about, since it spans densities where the minimiser
failed rather than a physical coexistence.

`maxwell_lines` assembles envelopes over temperatures into a table and brackets
triple points: a single-phase window X between neighbours L and R that exists at
one temperature while L and R coexist directly at the next. When the next
temperature's metastable L-X and X-R tangents are available, the signed window
width is interpolated linearly for an estimate of the triple temperature.

Notation: phi is the mean density, mu = df/dphi, omega = f - mu phi; `temp` is
the model's temperature-like parameter (epsilon for log/std PFC).
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

__all__ = [
    "STATUSES",
    "Segment",
    "Piece",
    "Branch",
    "TangentResult",
    "Window",
    "Envelope",
    "TripleBracket",
    "MaxwellLines",
    "analytic_branch",
    "tabulated_branch",
    "common_tangent",
    "lower_envelope",
    "maxwell_lines",
]

STATUSES = ("ok", "no_overlap", "no_root", "non_convex")

_RTOL = 4.0 * np.finfo(float).eps
_XTOL = 1e-15


# --------------------------------------------------------------------------- branches


@dataclass(frozen=True)
class Segment:
    """One contiguous density interval of a branch with its own representation.

    `f`, `mu`, `d2f` (may be None) and `q0star` (may be None) take and return
    numpy arrays and are only called inside ``[lo, hi]``. `nodes` holds the
    table densities of a tabulated segment (None for analytic).
    """

    lo: float
    hi: float
    f: Callable
    mu: Callable
    d2f: Callable | None = None
    q0star: Callable | None = None
    nodes: np.ndarray | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class Piece:
    """A convex part (f'' > 0) of one segment: mu is strictly increasing on it."""

    segment: int
    lo: float
    hi: float
    mu_lo: float
    mu_hi: float


@dataclass(frozen=True)
class Branch:
    """Free energy f(phi) of one phase over one or more density segments.

    Attributes
    ----------
    name : label used in results (phase name, e.g. ``"triangle"``).
    kind : ``"analytic"`` or ``"tabulated"``.
    segments : contiguous intervals (a tabulated branch has one per NaN-free run).
    pieces : convex parts of the segments, in density order; tangents use only these.
    nonconvex : ``(lo, hi)`` intervals where f'' <= 0, in density order.
    gaps : ``(lo, hi)`` density gaps between consecutive segments (NaN runs).
    dropped : table densities of finite runs too short to spline (< `min_points`).
    """

    name: str
    kind: str
    segments: tuple[Segment, ...]
    pieces: tuple[Piece, ...]
    nonconvex: tuple[tuple[float, float], ...]
    gaps: tuple[tuple[float, float], ...] = ()
    dropped: tuple[float, ...] = ()
    _samples: tuple = field(default=(), repr=False, compare=False)

    @property
    def lo(self) -> float:
        return self.segments[0].lo

    @property
    def hi(self) -> float:
        return self.segments[-1].hi

    @property
    def has_q0star(self) -> bool:
        return any(s.q0star is not None for s in self.segments)

    @property
    def mu_range(self) -> tuple[float, float]:
        """Sampled (min, max) of mu over all segments, concave parts included."""
        mu = self._samples[2]
        return float(np.min(mu)), float(np.max(mu))

    def _eval(self, attr: str, phi):
        x = np.asarray(phi, dtype=float)
        flat = np.atleast_1d(x).ravel()
        out = np.full(flat.shape, np.nan)
        for seg in self.segments:
            fn = getattr(seg, attr)
            if fn is None:
                continue
            m = (flat >= seg.lo) & (flat <= seg.hi)
            if m.any():
                out[m] = np.asarray(fn(flat[m]), dtype=float)
        if x.ndim == 0:
            return float(out[0])
        return out.reshape(x.shape)

    def f(self, phi):
        """f(phi); NaN outside the segments."""
        return self._eval("f", phi)

    def mu(self, phi):
        """mu(phi) = f'(phi); NaN outside the segments."""
        return self._eval("mu", phi)

    def q0star(self, phi):
        """q0*(phi) where a q0* representation was given; NaN otherwise."""
        return self._eval("q0star", phi)

    def piece_at(self, phi: float) -> int | None:
        """Index of the convex piece containing `phi`, or None."""
        for k, p in enumerate(self.pieces):
            if p.lo <= phi <= p.hi:
                return k
        return None


def _f2(x) -> float:
    return float(np.asarray(x, dtype=float))


def _make_samples(segments, n):
    """Dense samples (phi, f, mu, segment index) used for hull and line checks."""
    ph, ff, mm, ss = [], [], [], []
    for k, seg in enumerate(segments):
        x = np.linspace(seg.lo, seg.hi, n)
        if seg.nodes is not None:
            x = np.union1d(x, seg.nodes)
        ph.append(x)
        ff.append(np.asarray(seg.f(x), dtype=float))
        mm.append(np.asarray(seg.mu(x), dtype=float))
        ss.append(np.full(x.shape, k))
    return (np.concatenate(ph), np.concatenate(ff), np.concatenate(mm), np.concatenate(ss))


def _classify(segments, bounds_per_segment, convex_test):
    """Split segments at `bounds` into convex pieces and nonconvex intervals."""
    pieces, nonconvex = [], []
    for k, (seg, bounds) in enumerate(zip(segments, bounds_per_segment)):
        b = np.unique(np.clip(np.asarray(bounds, dtype=float), seg.lo, seg.hi))
        b = np.union1d(b, [seg.lo, seg.hi])
        tiny = 1e-12 * max(seg.hi - seg.lo, 1e-300)
        for u, v in zip(b[:-1], b[1:]):
            if v - u <= tiny:
                continue
            u, v = float(u), float(v)
            if convex_test(seg, u, v):
                mu_lo, mu_hi = _f2(seg.mu(np.array([u]))[0]), _f2(seg.mu(np.array([v]))[0])
                if pieces and pieces[-1].segment == k and pieces[-1].hi == u:
                    p = pieces.pop()  # adjacent convex intervals (f'' touches 0) merge
                    pieces.append(Piece(k, p.lo, v, p.mu_lo, mu_hi))
                else:
                    pieces.append(Piece(k, u, v, mu_lo, mu_hi))
            elif nonconvex and nonconvex[-1][1] == u:
                nonconvex[-1] = (nonconvex[-1][0], v)
            else:
                nonconvex.append((u, v))
    return tuple(pieces), tuple(nonconvex)


def analytic_branch(
    name: str,
    f: Callable,
    mu: Callable,
    lo: float,
    hi: float,
    *,
    d2f: Callable | None = None,
    q0star: Callable | None = None,
    n_sample: int = 2049,
) -> Branch:
    """Branch from vectorised callables on ``[lo, hi]`` (e.g. an adapter's liquid).

    `f` and `mu` (= df/dphi; the caller guarantees consistency) take numpy
    arrays. Convex pieces come from the sign of `d2f` when given; otherwise from
    the monotonicity of `mu` on `n_sample` points, with each turning point refined
    by a root of the central difference of `mu`. Pass ``functools.partial`` (or a
    lambda) to bind model parameters and temperature.
    """
    lo, hi = float(lo), float(hi)
    if not hi > lo:
        raise ValueError(f"analytic_branch {name!r}: need lo < hi, got {lo}, {hi}")
    seg = Segment(lo, hi, f, mu, d2f, q0star)
    x = np.linspace(lo, hi, n_sample)
    bounds = []
    if d2f is not None:
        s = np.asarray(d2f(x), dtype=float)
        for i in np.nonzero(np.sign(s[:-1]) != np.sign(s[1:]))[0]:
            u, v = x[i], x[i + 1]
            try:
                bounds.append(brentq(lambda p: _f2(d2f(np.array([p]))[0]), u, v,
                                     xtol=_XTOL, rtol=_RTOL))
            except ValueError:
                bounds.append(0.5 * (u + v))

        def convex(sg, u, v):
            return _f2(sg.d2f(np.array([0.5 * (u + v)]))[0]) > 0.0
    else:
        m = np.asarray(mu(x), dtype=float)
        dm = np.sign(np.diff(m))
        h = 1e-6 * max(1.0, abs(lo), abs(hi))

        def d2num(p):
            a, b = max(lo, p - h), min(hi, p + h)
            return (_f2(mu(np.array([b]))[0]) - _f2(mu(np.array([a]))[0])) / (b - a)

        for i in np.nonzero(dm[:-1] != dm[1:])[0]:
            u, v = x[i], x[i + 2]
            try:
                bounds.append(brentq(d2num, u, v, xtol=1e-14, rtol=_RTOL))
            except ValueError:
                bounds.append(x[i + 1])

        def convex(sg, u, v):
            return _f2(sg.mu(np.array([v]))[0]) > _f2(sg.mu(np.array([u]))[0])

    pieces, nonconvex = _classify((seg,), [bounds], convex)
    return Branch(name, "analytic", (seg,), pieces, nonconvex,
                  _samples=_make_samples((seg,), n_sample))


def tabulated_branch(
    name: str,
    phi,
    f,
    q0star=None,
    *,
    min_points: int = 4,
    bc_type="not-a-knot",
    n_sample: int = 1025,
) -> Branch:
    """Branch from a table f(phi) (e.g. a q0* scan's fmin), NaN where not converged.

    Points are sorted by phi; a point is usable when phi and f (and q0star, when
    given) are finite. Each maximal run of usable consecutive points becomes its
    own `Segment` with a `CubicSpline` of f (mu = its derivative, f'' its second
    derivative) and, when `q0star` is given, a `CubicSpline` of q0* with the same
    end condition. Runs shorter than `min_points` are dropped and listed in
    `Branch.dropped`. All segments stay in one branch; the NaN runs between them
    are `Branch.gaps`, and a tangent between two segments is flagged `across_gap`.
    Concave intervals are exact: f'' of a cubic spline is piecewise linear.
    """
    phi = np.asarray(phi, dtype=float).ravel()
    f = np.asarray(f, dtype=float).ravel()
    q = None if q0star is None else np.asarray(q0star, dtype=float).ravel()
    if phi.shape != f.shape or (q is not None and q.shape != phi.shape):
        raise ValueError(f"tabulated_branch {name!r}: phi, f, q0star must have equal length")
    order = np.argsort(phi, kind="stable")
    phi, f = phi[order], f[order]
    q = None if q is None else q[order]
    good = np.isfinite(phi) & np.isfinite(f)
    if q is not None:
        good &= np.isfinite(q)
    if np.any(np.diff(phi[np.isfinite(phi)]) == 0):
        raise ValueError(f"tabulated_branch {name!r}: repeated phi values")
    runs, start = [], None
    for i, g_ in enumerate(good):
        if g_ and start is None:
            start = i
        if not g_ and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(good)))
    segments, dropped, d2s = [], [], []
    for i, j in runs:
        if j - i < min_points:
            dropped.extend(float(v) for v in phi[i:j])
            continue
        x = phi[i:j]
        spl = CubicSpline(x, f[i:j], bc_type=bc_type)
        qs = None if q is None else CubicSpline(x, q[i:j], bc_type=bc_type)
        d2 = spl.derivative(2)
        segments.append(Segment(float(x[0]), float(x[-1]), spl, spl.derivative(1), d2, qs, x))
        d2s.append(d2)
    if not segments:
        raise ValueError(f"tabulated_branch {name!r}: no run of >= {min_points} finite points")
    bounds = []
    for d2 in d2s:
        r = np.asarray(d2.roots(extrapolate=False), dtype=float)
        bounds.append(r[np.isfinite(r)])

    def convex(sg, u, v):
        return _f2(sg.d2f(0.5 * (u + v))) > 0.0

    pieces, nonconvex = _classify(segments, bounds, convex)
    gaps = tuple((segments[k].hi, segments[k + 1].lo) for k in range(len(segments) - 1))
    return Branch(name, "tabulated", tuple(segments), pieces, nonconvex, gaps, tuple(dropped),
                  _samples=_make_samples(segments, n_sample))


# ----------------------------------------------------------------- pairwise tangent


@dataclass(frozen=True)
class TangentResult:
    """Common tangent between branches `a` and `b` (names), or why there is none.

    phi_a, phi_b are the coexisting densities on `a` and `b` in argument order
    (either may be the lower one). residual_mu_a/b = mu(phi) - mu on each side and
    residual_omega = omega_a - omega_b. piece_a/piece_b index `Branch.pieces`.
    `supporting` is True when neither branch dips below the tangent line by more
    than `atol`. `stable` and `below` are set by `lower_envelope`: True for a
    bridge of the envelope, False for a metastable tangent, with `below` naming a
    branch that lies under its line. `candidates` holds every tangent found (the
    primary included) when the search was not restricted.
    """

    a: str
    b: str
    status: str
    phi_a: float = math.nan
    phi_b: float = math.nan
    mu: float = math.nan
    omega: float = math.nan
    q0star_a: float | None = None
    q0star_b: float | None = None
    residual_mu_a: float = math.nan
    residual_mu_b: float = math.nan
    residual_omega: float = math.nan
    piece_a: int | None = None
    piece_b: int | None = None
    same_branch: bool = False
    across_gap: bool = False
    supporting: bool | None = None
    stable: bool | None = None
    below: str | None = None
    message: str = ""
    candidates: tuple = field(default=(), repr=False)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def residual(self) -> float:
        """Largest absolute residual (NaN unless ok)."""
        return max(abs(self.residual_mu_a), abs(self.residual_mu_b), abs(self.residual_omega))

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("candidates")
        return d


def _invert(br: Branch, k: int, mu: float) -> float:
    p = br.pieces[k]
    if mu <= p.mu_lo:
        return p.lo
    if mu >= p.mu_hi:
        return p.hi
    fn = br.segments[p.segment].mu
    return brentq(lambda x: _f2(fn(x)) - mu, p.lo, p.hi, xtol=_XTOL, rtol=_RTOL)


def _omega(br: Branch, k: int, mu: float) -> tuple[float, float]:
    x = _invert(br, k, mu)
    return _f2(br.segments[br.pieces[k].segment].f(x)) - mu * x, x


def _domega(a, i, b, j, mu):
    return _omega(a, i, mu)[0] - _omega(b, j, mu)[0]


def _roots(fn, lo, hi, n):
    m = np.linspace(lo, hi, max(int(n), 2))
    v = np.array([fn(x) for x in m])
    out = []
    for k in range(len(m)):
        if v[k] == 0.0:
            out.append(float(m[k]))
        elif k + 1 < len(m) and v[k] * v[k + 1] < 0.0:
            out.append(brentq(fn, m[k], m[k + 1], xtol=_XTOL, rtol=_RTOL))
    out.sort()
    uniq = []
    for r in out:
        if not uniq or abs(r - uniq[-1]) > 1e-12 * (1.0 + abs(r)):
            uniq.append(r)
    return uniq


def _line_gap(br: Branch, mu: float, phi0: float, f0: float):
    """Minimum over the branch samples of f - tangent line, and the phi where it occurs."""
    ph, ff, _, ss = br._samples
    gap = ff - (f0 + mu * (ph - phi0))
    k = int(np.argmin(gap))
    return float(gap[k]), float(ph[k])


def _candidate(a, i, b, j, mu, same, atol):
    sa, sb = a.segments[a.pieces[i].segment], b.segments[b.pieces[j].segment]
    om_a, xa = _omega(a, i, mu)
    om_b, xb = _omega(b, j, mu)
    fa = om_a + mu * xa
    gap = min(_line_gap(a, mu, xa, fa)[0], _line_gap(b, mu, xa, fa)[0])
    return TangentResult(
        a=a.name, b=b.name, status="ok", phi_a=float(xa), phi_b=float(xb), mu=float(mu),
        omega=float(om_a),
        q0star_a=None if sa.q0star is None else _f2(sa.q0star(xa)),
        q0star_b=None if sb.q0star is None else _f2(sb.q0star(xb)),
        residual_mu_a=_f2(sa.mu(xa)) - mu, residual_mu_b=_f2(sb.mu(xb)) - mu,
        residual_omega=float(om_a - om_b), piece_a=i, piece_b=j, same_branch=same,
        across_gap=bool(same and a.pieces[i].segment != b.pieces[j].segment),
        supporting=bool(gap >= -atol),
    )


def common_tangent(
    a: Branch,
    b: Branch,
    *,
    pieces: tuple[int, int] | None = None,
    ordered: bool = False,
    mu_guess: float | None = None,
    n_scan: int = 33,
    atol: float = 1e-10,
) -> TangentResult:
    """Common tangent (equal mu and omega = f - mu phi) between branches a and b.

    Every pair of convex pieces (one of `a`, one of `b`; distinct pieces when
    ``a is b``, which gives a self-tangent across a concave interval) is searched
    on the overlap of their mu ranges: omega_a - omega_b is sampled at `n_scan`
    points and each sign change refined by brentq. Options:

    * `pieces` - restrict to one pair ``(index into a.pieces, index into b.pieces)``.
    * `ordered` - accept only tangents with phi_a < phi_b.
    * `mu_guess` - among several candidates prefer the one nearest this mu.

    Primary selection among candidates: supporting ones first (no branch of the
    pair below the line), then nearest `mu_guess`, else the lowest mu; all are
    in `.candidates`. Never raises for a pair without a tangent: see `STATUSES`
    in the module docstring for the status rules.
    """
    same = a is b
    for br in (a, b):
        if not br.pieces:
            return TangentResult(a.name, b.name, "non_convex", same_branch=same,
                                 message=f"branch {br.name!r} has no convex piece")
    if pieces is not None:
        pairs = [tuple(pieces)]
    elif same:
        n = len(a.pieces)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        pairs = [(i, j) for i in range(len(a.pieces)) for j in range(len(b.pieces))]
    cands, overlap = [], False
    for i, j in pairs:
        pa, pb = a.pieces[i], b.pieces[j]
        lo, hi = max(pa.mu_lo, pb.mu_lo), min(pa.mu_hi, pb.mu_hi)
        if not lo < hi:
            continue
        overlap = True
        for m in _roots(lambda x: _domega(a, i, b, j, x), lo, hi, n_scan):
            cands.append(_candidate(a, i, b, j, m, same, atol))
    if ordered:
        cands = [c for c in cands if c.phi_a < c.phi_b]
    if cands:
        def key(c):
            near = abs(c.mu - mu_guess) if mu_guess is not None else c.mu
            return (not c.supporting, near)

        cands.sort(key=key)
        best = replace(cands[0], candidates=tuple(cands) if pieces is None else ())
        if len(cands) > 1:
            best = replace(best, message=f"{len(cands)} tangents found; primary chosen by "
                                          "supporting, then mu_guess/lowest mu")
        return best
    if same and len(a.pieces) < 2:
        return TangentResult(a.name, b.name, "no_root", same_branch=True,
                             message="branch is convex on one piece; no self-tangent")
    (alo, ahi), (blo, bhi) = a.mu_range, b.mu_range
    if not max(alo, blo) < min(ahi, bhi):
        return TangentResult(a.name, b.name, "no_overlap", same_branch=same,
                             message=f"mu ranges [{alo:.6g}, {ahi:.6g}] and "
                                     f"[{blo:.6g}, {bhi:.6g}] are disjoint")
    if overlap:
        msg = "no sign change of omega_a - omega_b on the convex mu overlap"
        if ordered:
            msg += " with phi_a < phi_b"
        return TangentResult(a.name, b.name, "no_root", same_branch=same, message=msg)
    return TangentResult(a.name, b.name, "non_convex", same_branch=same,
                         message="mu ranges overlap only through concave parts")


# ------------------------------------------------------------------- lower envelope


@dataclass(frozen=True)
class Window:
    """Single-phase window of the envelope: densities where `name` alone is stable."""

    name: str
    branch: int
    piece: int
    phi_lo: float
    phi_hi: float

    @property
    def width(self) -> float:
        return self.phi_hi - self.phi_lo


@dataclass(frozen=True)
class Envelope:
    """Lower envelope of all branches at one temperature.

    `windows` and `coexistence` alternate in density order: windows[k] |
    coexistence[k] | windows[k+1]. `metastable` holds every other pairwise
    tangent found, each with `stable=False` and `below` naming a branch under
    its line. `hull_phi`, `hull_f` are the sampled hull vertices (for plots).
    """

    temp: float | None
    names: tuple[str, ...]
    windows: tuple[Window, ...]
    coexistence: tuple[TangentResult, ...]
    metastable: tuple[TangentResult, ...]
    warnings: tuple[str, ...]
    hull_phi: np.ndarray = field(repr=False, compare=False, default=None)
    hull_f: np.ndarray = field(repr=False, compare=False, default=None)


def _lower_hull(x, y):
    """Indices of the lower convex hull of points sorted by x (monotone chain)."""
    h = []
    for k in range(len(x)):
        while len(h) >= 2:
            o, p = h[-2], h[-1]
            if (x[p] - x[o]) * (y[k] - y[o]) - (y[p] - y[o]) * (x[k] - x[o]) <= 0.0:
                h.pop()
            else:
                break
        h.append(k)
    return h


def _nearest_piece(br: Branch, phi: float, seg: int) -> int:
    best, dist = -1, math.inf
    for k, p in enumerate(br.pieces):
        if p.segment != seg:
            continue
        d = 0.0 if p.lo <= phi <= p.hi else min(abs(phi - p.lo), abs(phi - p.hi))
        if d < dist:
            best, dist = k, d
    return best


def _grid_samples(br: Branch, grid):
    ph, ff, ss = [], [], []
    for k, seg in enumerate(br.segments):
        x = grid[(grid >= seg.lo) & (grid <= seg.hi)]
        x = np.union1d(x, [seg.lo, seg.hi])
        ph.append(x)
        ff.append(np.asarray(seg.f(x), dtype=float))
        ss.append(np.full(x.shape, k))
    return np.concatenate(ph), np.concatenate(ff), np.concatenate(ss)


def lower_envelope(
    branches: Sequence[Branch],
    phi_grid=None,
    *,
    temp: float | None = None,
    n: int = 4001,
    n_scan: int = 33,
    atol: float = 1e-10,
    max_iter: int | None = None,
) -> Envelope:
    """Coexisting pairs and single-phase windows of all branches at one temperature.

    The lower convex hull of all branches sampled on `phi_grid` (default: `n`
    uniform points over the union of their domains, plus segment ends) gives the
    sequence of single-phase stretches; consecutive stretches on different
    branches (or different pieces of one branch) are bridged by
    ``common_tangent(left, right, pieces=..., ordered=True)``. The refined
    bridges are then checked exactly against every branch's dense samples: a
    window whose bridges cross (negative width) is removed, and a branch dipping
    below a bridge's line by more than `atol` is inserted as a window, until
    nothing changes. All other pairwise tangents (all candidates of every pair,
    and self-tangents of branches with several pieces) are returned as
    metastable. Branch names must be unique.
    """
    branches = list(branches)
    if not branches:
        raise ValueError("lower_envelope: no branches")
    names = tuple(b.name for b in branches)
    if len(set(names)) != len(names):
        raise ValueError(f"lower_envelope: branch names must be unique, got {names}")
    warns: list[str] = []
    if phi_grid is None:
        grid = np.linspace(min(b.lo for b in branches), max(b.hi for b in branches), n)
    else:
        grid = np.asarray(phi_grid, dtype=float)
    P, F, LB, LP = [], [], [], []
    for bi, br in enumerate(branches):
        ph, ff, ss = _grid_samples(br, grid)
        pc = np.array([_nearest_piece(br, x, s) for x, s in zip(ph, ss)], dtype=int)
        P.append(ph)
        F.append(ff)
        LB.append(np.full(ph.shape, bi))
        LP.append(pc)
    P, F, LB, LP = map(np.concatenate, (P, F, LB, LP))
    order = np.lexsort((F, P))
    P, F, LB, LP = P[order], F[order], LB[order], LP[order]
    keep = np.ones(P.shape, bool)
    keep[1:] = P[1:] != P[:-1]  # equal phi: lowest f first after lexsort
    P, F, LB, LP = P[keep], F[keep], LB[keep], LP[keep]
    hull = _lower_hull(P, F)
    groups: list[tuple[int, int]] = []
    guesses: dict = {}
    for k in hull:
        lab = (int(LB[k]), int(LP[k]))
        if lab[1] < 0:
            warns.append(f"hull vertex at phi={P[k]:.6g} on concave-only segment of "
                         f"{names[lab[0]]!r} ignored")
            continue
        if groups and groups[-1] == lab:
            continue
        if groups:
            prev = hull[hull.index(k) - 1]
            guesses[(groups[-1], lab)] = (F[k] - F[prev]) / (P[k] - P[prev])
        groups.append(lab)

    cache: dict = {}

    def bridge(gl, gr):
        if (gl, gr) not in cache:
            a, b = branches[gl[0]], branches[gr[0]]
            cache[(gl, gr)] = common_tangent(a, b, pieces=(gl[1], gr[1]), ordered=True,
                                             mu_guess=guesses.get((gl, gr)), n_scan=n_scan,
                                             atol=atol)
        return cache[(gl, gr)]

    def deepest(res):
        best = (math.inf, None, None)
        f0 = res.omega + res.mu * res.phi_a
        for bi, br in enumerate(branches):
            gap, x = _line_gap(br, res.mu, res.phi_a, f0)
            if gap < best[0]:
                best = (gap, bi, x)
        return best

    limit = max_iter if max_iter is not None else 4 * len(groups) + 4 * len(branches) + 10
    stuck: set = set()
    for _ in range(limit):
        merged = [g for k, g in enumerate(groups) if k == 0 or g != groups[k - 1]]
        groups = merged
        bridges = [bridge(groups[k], groups[k + 1]) for k in range(len(groups) - 1)]
        changed = False
        for k in range(1, len(groups) - 1):
            L, R = bridges[k - 1], bridges[k]
            if L.ok and R.ok and L.phi_b > R.phi_a:
                del groups[k]
                changed = True
                break
        if changed:
            continue
        for k, br in enumerate(bridges):
            if not br.ok:
                continue
            gap, bi, x = deepest(br)
            if gap < -atol:
                seg = int(np.argmin([0 if s.lo <= x <= s.hi else 1
                                     for s in branches[bi].segments]))
                lab = (bi, _nearest_piece(branches[bi], x, seg))
                if lab[1] < 0 or lab in (groups[k], groups[k + 1]):
                    if (k, lab) not in stuck:
                        stuck.add((k, lab))
                        warns.append(f"{names[bi]!r} dips {gap:.3g} below the "
                                     f"{br.a}-{br.b} tangent at phi={x:.6g} but cannot be "
                                     "inserted (same piece as a neighbour or concave)")
                    continue
                groups.insert(k + 1, lab)
                changed = True
                break
        if not changed:
            break
    else:
        warns.append("envelope refinement did not converge; result may be inconsistent")
    groups = [g for k, g in enumerate(groups) if k == 0 or g != groups[k - 1]]
    bridges = [bridge(groups[k], groups[k + 1]) for k in range(len(groups) - 1)]
    bridges = [replace(b_, stable=True) if b_.ok else b_ for b_ in bridges]
    windows = []
    for k, (bi, pi) in enumerate(groups):
        p = branches[bi].pieces[pi]
        lo = bridges[k - 1].phi_b if k > 0 and bridges[k - 1].ok else p.lo
        hi = bridges[k].phi_a if k < len(bridges) and bridges[k].ok else p.hi
        windows.append(Window(names[bi], bi, pi, float(lo), float(hi)))
    for br in bridges:
        if not br.ok:
            warns.append(f"hull bridge {br.a}-{br.b} has no tangent ({br.status}: "
                         f"{br.message}); the envelope there is domain-limited")
        elif br.across_gap:
            warns.append(f"bridge {br.a}-{br.b} spans a NaN gap of {br.a!r} "
                         f"(phi {br.phi_a:.6g} to {br.phi_b:.6g})")

    stable_keys = []
    for br in bridges:
        if br.ok:
            stable_keys.append((br.a, br.piece_a, br.b, br.piece_b, br.mu))
            stable_keys.append((br.b, br.piece_b, br.a, br.piece_a, br.mu))

    def is_stable(c):
        return any(c.a == s[0] and c.piece_a == s[1] and c.b == s[2] and c.piece_b == s[3]
                   and abs(c.mu - s[4]) <= 1e-9 * (1.0 + abs(c.mu)) for s in stable_keys)

    meta = []
    for i in range(len(branches)):
        for j in range(i, len(branches)):
            if i == j and len(branches[i].pieces) < 2:
                continue
            res = common_tangent(branches[i], branches[j], n_scan=n_scan, atol=atol)
            for c in (res.candidates or ((res,) if res.ok else ())):
                if is_stable(c):
                    continue
                gap, bi, x = deepest(c)
                if gap < -atol:
                    meta.append(replace(c, stable=False, below=names[bi], candidates=()))
                else:
                    meta.append(replace(c, stable=False, candidates=()))
                    warns.append(f"tangent {c.a}-{c.b} at mu={c.mu:.6g} supports every "
                                 "branch but is not on the sampled hull; refine phi_grid")
    meta.sort(key=lambda c: c.mu)
    hv = np.array(hull, dtype=int)
    return Envelope(temp, names, tuple(windows), tuple(bridges), tuple(meta), tuple(warns),
                    P[hv], F[hv])


# -------------------------------------------------------------------- Maxwell lines


@dataclass(frozen=True)
class TripleBracket:
    """Triple point L-X-R bracketed between two consecutive temperatures.

    The window of X between L and R exists at `temp_open` and is closed (L and
    R coexist directly) at `temp_closed`. `width_open` is that window's width;
    `width_closed` the signed width from the metastable L-X and X-R tangents at
    `temp_closed` (negative), when both exist, in which case `temp_estimate` is
    the linear zero of the width.
    """

    phases: tuple[str, str, str]
    temp_lo: float
    temp_hi: float
    temp_open: float
    temp_closed: float
    width_open: float
    width_closed: float | None = None
    temp_estimate: float | None = None


_ROW_KEYS = ("temp", "a", "b", "phi_a", "phi_b", "mu", "omega", "q0star_a", "q0star_b",
             "status", "same_branch", "across_gap", "stable", "below", "residual")


def _row(t, c):
    d = {k: getattr(c, k) for k in _ROW_KEYS if k not in ("temp", "residual")}
    d["temp"] = t
    d["residual"] = c.residual if c.ok else math.nan
    return {k: d[k] for k in _ROW_KEYS}


@dataclass(frozen=True)
class MaxwellLines:
    """Coexistence lines over temperatures.

    `rows`: one dict per stable coexistence (keys `_ROW_KEYS`), by temperature then
    density; `metastable`: same keys for metastable tangents; `windows`: dicts
    (temp, name, phi_lo, phi_hi); `triple_points`: `TripleBracket` list.
    """

    rows: tuple[dict, ...]
    metastable: tuple[dict, ...]
    windows: tuple[dict, ...]
    triple_points: tuple[TripleBracket, ...]
    temps: tuple[float, ...]

    def columns(self, which: str = "rows") -> dict:
        """Column-oriented copy of `rows`, `metastable` or `windows`."""
        data = getattr(self, which)
        keys = list(data[0].keys()) if data else (list(_ROW_KEYS) if which != "windows"
                                                  else ["temp", "name", "phi_lo", "phi_hi"])
        return {k: [r[k] for r in data] for k in keys}

    def write_csv(self, dest, which: str = "rows") -> None:
        """Write `rows`, `metastable` or `windows` as CSV (repr floats, '' for None).

        `dest` is a path (written as ASCII with LF line ends) or an open text file.
        """
        keys = list(self.columns(which))

        def emit(fh):
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(keys)
            for r in getattr(self, which):
                w.writerow(["" if r[k] is None else (repr(r[k]) if isinstance(r[k], float)
                                                      else r[k]) for k in keys])

        if hasattr(dest, "write"):
            emit(dest)
        else:
            with open(dest, "w", newline="", encoding="ascii") as fh:
                emit(fh)


def _signed_width(env: Envelope, L: str, X: str, R: str, mu_ref: float | None):
    def side(c, left, right):
        if c.a == left and c.b == right and c.phi_a < c.phi_b:
            return c.phi_b if right == X else c.phi_a
        if c.a == right and c.b == left and c.phi_b < c.phi_a:
            return c.phi_a if right == X else c.phi_b
        return None

    lx, xr = [], []
    for c in env.metastable:
        if not c.ok:
            continue
        v = side(c, L, X)
        if v is not None:
            lx.append((c.mu, v))
        v = side(c, X, R)
        if v is not None:
            xr.append((c.mu, v))
    if not lx or not xr:
        return None
    pick = (lambda s: min(s, key=lambda t: abs(t[0] - mu_ref))) if mu_ref is not None else \
        (lambda s: s[0])
    return pick(xr)[1] - pick(lx)[1]


def maxwell_lines(per_temperature_results: Mapping | Iterable[Envelope]) -> MaxwellLines:
    """Assemble `lower_envelope` results over temperatures; bracket triple points.

    Accepts ``{temp: Envelope}`` or an iterable of envelopes built with `temp`
    set. Window sequences (branch names in density order) of consecutive
    temperatures are compared: a window X with neighbours L, R at one temperature
    whose L and R are adjacent at the other gives a `TripleBracket`.
    """
    if isinstance(per_temperature_results, Mapping):
        items = [(float(t), e) for t, e in per_temperature_results.items()]
    else:
        items = []
        for e in per_temperature_results:
            if e.temp is None:
                raise ValueError("maxwell_lines: envelope without temp; pass a mapping")
            items.append((float(e.temp), e))
    items.sort(key=lambda te: te[0])
    rows, meta, wins = [], [], []
    for t, e in items:
        rows.extend(_row(t, c) for c in e.coexistence)
        meta.extend(_row(t, c) for c in e.metastable)
        wins.extend({"temp": t, "name": w.name, "phi_lo": w.phi_lo, "phi_hi": w.phi_hi}
                    for w in e.windows)
    triples = []
    for (t1, e1), (t2, e2) in zip(items[:-1], items[1:]):
        for (ta, ea), (tb, eb) in (((t1, e1), (t2, e2)), ((t2, e2), (t1, e1))):
            sa = [w.name for w in ea.windows]
            sb = [w.name for w in eb.windows]
            for k in range(1, len(sa) - 1):
                L, X, R = sa[k - 1], sa[k], sa[k + 1]
                adj = [m for m in range(len(sb) - 1) if sb[m] == L and sb[m + 1] == R]
                kept = any(sb[m - 1:m + 2] == [L, X, R] for m in range(1, len(sb) - 1))
                if not adj or kept:
                    continue
                w_open = ea.windows[k].width
                mu_ref = eb.coexistence[adj[0]].mu if eb.coexistence[adj[0]].ok else None
                w_closed = _signed_width(eb, L, X, R, mu_ref)
                est = None
                if w_closed is not None and w_closed < 0.0 < w_open:
                    est = ta + (tb - ta) * w_open / (w_open - w_closed)
                triples.append(TripleBracket((L, X, R), min(ta, tb), max(ta, tb), ta, tb,
                                             float(w_open), w_closed, est))
    return MaxwellLines(tuple(rows), tuple(meta), tuple(wins), tuple(triples),
                        tuple(t for t, _ in items))
