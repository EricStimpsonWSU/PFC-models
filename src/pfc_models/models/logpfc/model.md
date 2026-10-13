# log-PFC, 2D spectral, GPU

**Status:** staged for review (2026-10-12) · **Declared features:** see [model.toml](model.toml)

## Free energy

The free-energy *density* as implemented in `Simulation.CalculateFreeEnergy`:

$$
f[\phi] \;=\; \phi\,\frac{\beta}{2}\bigl(q_0^2 + \nabla^2\bigr)^2 \phi
\;+\; \phi\ln\phi \;+\; \frac{\varepsilon}{2}\phi^2
\;+\; \frac{g}{3}\phi^3 \;+\; \frac{1}{4}\phi^4
$$

`CalculateFreeEnergy` returns the mean of $f$ over the box.

| symbol | code | meaning |
|---|---|---|
| $\phi$ | `State.phi` | dimensionless density field |
| $\beta$ | `Model.beta` | stiffness of the gradient (peak-selection) term |
| $q_0$ | `Model.q0` | reference wavenumber selecting the lattice spacing |
| $\varepsilon$ | `Model.epsilon` | quadratic coefficient (temperature-like) |
| $g$ | `Model.g` | cubic coefficient |
| $\Gamma$ | `Model.Gamma` | mobility in the equation of motion |

Two notes on the implementation, neither of them a change made during migration:

* the ideal-gas term is written $\phi\ln\phi$ rather than the more usual
  $\phi\ln\phi - \phi$;
* the chemical potential below carries $\ln\phi$ where
  $\partial(\phi\ln\phi)/\partial\phi = \ln\phi + 1$. The constant is immaterial
  because the dynamics use $\nabla^2\mu$.

## Equations of motion

Conserved (Cahn-Hilliard) dynamics driven by the chemical potential:

$$
\frac{\partial \phi}{\partial t} \;=\; \Gamma \nabla^2 \mu,
\qquad
\mu \;=\; \ln\phi + g\phi^2 + \phi^3
      + \Bigl[\beta\bigl(q_0^2 + \nabla^2\bigr)^2 + \varepsilon\Bigr]\phi
$$

Split for time stepping into a linear operator and a nonlinear remainder,

$$
\hat{L} = \Gamma k^2_{\text{lap}} \Bigl[\beta\bigl(q_0^2 + k^2_{\text{lap}}\bigr)^2
          + \varepsilon\Bigr],
\qquad
\hat{N} = \Gamma k^2_{\text{lap}} \,\widehat{\bigl(\ln\phi + g\phi^2 + \phi^3\bigr)}
$$

with $k^2_{\text{lap}} = -(k_x^2 + k_y^2)$, and advanced by first-order
exponential time differencing (ETD1):

$$
\hat\phi(t + \Delta t) \;=\; e^{\hat{L}\Delta t}\,\hat\phi(t)
   \;+\; \varphi_1(\hat{L}\Delta t)\,\hat{N}(t),
\qquad
\varphi_1(z) = \frac{e^{z} - 1}{\hat{L}}
$$

`kernel_etd` in the spectral substrate returns the two coefficients. Near
$z = \hat{L}\Delta t \to 0$ the quotient is a removable singularity, so below
`small_z_tol` it uses the Taylor form
$\Delta t\,(1 + z/2 + z^2/6)$.

## Assumptions

* Two dimensions, fully periodic, uniform rectangular grid.
* Spectral spatial discretisation: all derivatives are Fourier multipliers, so
  the solution is assumed smooth and band-limited on the grid.
* `float64` throughout.
* Mass is conserved: the dynamics are of divergence form and the `k = 0` mode of
  the update carries no nonlinear contribution.
* The model cannot represent free surfaces, non-periodic boundaries, or any
  dimensionality other than 2 - those are separate catalogue entries by design.

## Numerical notes

### Clamping of log(phi)

The correct form clamps the logarithm only:

```python
nonlin_mu = where(phi > minPhi, log(phi), minLog) + g * phi**2 + phi**3
```

Clamping the full expression (`where(phi > minPhi, log(phi) + g*phi**2 + phi**3,
minLog)`) is an implementation bug: it discards the polynomial terms wherever
`phi <= minPhi`. No fork in this migration carried the buggy form.

**This migration applied no clamp change.** This fork computes the
unclamped expression and carries the clamped line as a comment; both are
relocated verbatim. Clamping is deliberately deferred pending a conditional
implementation, so the migrated model reproduces the source fork exactly.

### The small-z threshold

`GetExponentialTimesteppingKernels` had four bodies across the source tree,
disagreeing on the guard against the removable singularity in
`(exp(z) - 1)/kLinear`. They are reconciled into one substrate function,
`kernel_etd`, whose threshold is the keyword `small_z_tol` rather than a
hard-coded constant, settable per run through ``BuildLogPFCModelParameters(..., smallZTol=...)``.
This model declares **`smallZTol = 2e-5`**, the value its source fork used.

### Initial-condition seeding

The two generators in `pfc_models.fields.initial_conditions_2d` that draw
randomness (`GenerateHotInitialConditions`,
`GenerateLogNormalHotInitialConditions`) now take their stream from
`np.random.default_rng(seed)` instead of calling `np.random.seed(seed)` on
global NumPy state. The `seed=None` default and the call signature are unchanged
from the source, so no call site moved - but `default_rng(n)` and
`np.random.seed(n)` produce different streams, so a run seeded under the source
fork does not reproduce here.

## Provenance

Migrated from `log/gpu-spectral-base-implementation` (2026-10-12), staged under
`sandbox/log-pfc-migration/staged/`. Relocation only: `Geometry`,
`Build2DGeometry`, `_DerivativeKernels`, `BuildDerivativeKernels`,
`Get_kGaussian` and `GetExponentialTimesteppingKernels` moved to the spectral
substrate, imports rewritten, two call sites renamed, `smallZTol` threaded
through the parameter builder. No physics changed.

Full per-file record: [migration-log.md](../../../../../migration-log.md).

Migration checklist:

- [x] substrate dependencies available (`geometry_2d`, `kernels`, `backends`)
- [x] clamp decision recorded (no change applied this step)
- [x] initial conditions draw from an explicit Generator
- [ ] `test_regression.py` written and passing - **omitted by decision for this migration step**; behaviour is verified visually from the example notebooks
- [ ] `model.toml` `migrated = true`
- [ ] source directory recorded as superseded in the old repo

## Results that used this model

_Paper, run-id, tag._
