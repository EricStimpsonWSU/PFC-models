# log-PFC with external potential (vacancy forcing)

**Status:** staged for review (2026-10-12), **migrated in reduced form** · **Declared features:** see [model.toml](model.toml)

> This model is a **reduction** of its source fork. The finite-difference
> Laplacian and the predictor-corrector stepper are deferred for a later
> refactor; both original expressions are quoted below so the refactor has an
> exact target. Re-running the example notebooks will not reproduce the output
> the source fork produced.

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

Conserved dynamics with an external potential $V$ and a vacancy-forcing field
$M$:

$$
\frac{\partial \phi}{\partial t} \;=\; \Gamma \nabla^2 \mu \;+\; \nabla^2 V,
\qquad
\mu \;=\; \ln\phi + g\phi^2 + \phi^3 - M\phi
      + \Bigl[\beta\bigl(q_0^2 + \nabla^2\bigr)^2 + \varepsilon + M\Bigr]\phi
$$

$M$ is set by `Simulation.OptimizeM` to keep $\phi$ inside its validity range
and enters both the nonlinear and the linear part of $\mu$; $V$ is built by
`external_potentials` and enters the nonlinear force as
$\nabla^2 \hat{V}$. Advanced by ETD1 exactly as for `logpfc`, with the
nonlinear term carrying the external-potential force:

```python
phi_next_hat = kernel_linexp_phidt * phi_hat + kernel_nonlinexp_phidt * (
    Gamma * kernel_laplacian * nonlin_mu_hat + kernel_laplacian * externalV_hat
)
phi_next_hat.flat[0] = phi_hat.flat[0]  # enforce mass conservation
```

## Assumptions

* Two dimensions, fully periodic, uniform rectangular grid.
* Spectral spatial discretisation: all derivatives are Fourier multipliers, so
  the solution is assumed smooth and band-limited on the grid.
* `float64` throughout.
* Mass is conserved: the dynamics are of divergence form and the `k = 0` mode of
  the update carries no nonlinear contribution.
* The model cannot represent free surfaces, non-periodic boundaries, or any
  dimensionality other than 2 - those are separate catalogue entries by design.

* `Geometry` additionally carries the triangular-lattice constants `a` and `h`,
  attached by this model's `BuildGeometry` wrapper rather than by the shared
  substrate builder. Use `BuildGeometry`, not `Build2DGeometry`, or `OptimizeM`
  and `GetForce` will not find them.

## Numerical notes

### Clamping of log(phi)

The correct form clamps the logarithm only:

```python
nonlin_mu = where(phi > minPhi, log(phi), minLog) + g * phi**2 + phi**3
```

Clamping the full expression (`where(phi > minPhi, log(phi) + g*phi**2 + phi**3,
minLog)`) is an implementation bug: it discards the polynomial terms wherever
`phi <= minPhi`. No fork in this migration carried the buggy form.

**This migration applied no clamp change.** This fork already clamps the
logarithm alone, in the correct form, and additionally subtracts the
vacancy-forcing term:

```python
nonlin_mu[...] = where(phi > minPhi, log(phi), minLog) + g*phi**2 + phi**3 - M*phi
```

It is relocated verbatim. It is the reference form for the other two models when
clamping is implemented there.

### The small-z threshold

`GetExponentialTimesteppingKernels` had four bodies across the source tree,
disagreeing on the guard against the removable singularity in
`(exp(z) - 1)/kLinear`. They are reconciled into one substrate function,
`kernel_etd`, whose threshold is the keyword `small_z_tol` rather than a
hard-coded constant, settable per run through ``BuildLogPFCModel_withVacancyForce_Parameters(..., smallZTol=...)``.
This model declares **`smallZTol = 1e-8`**, the value its source fork used.

### Deferred: the finite-difference Laplacian

The source fork had replaced the spectral Laplacian with the Fourier symbol of
the five-point finite-difference Laplacian, leaving the spectral form commented
out:

```python
# # Spectral laplacian:
# derivativeKernels.d2_dlap = -(geometry.KX**2 + geometry.KY**2)

# Finite differences laplacian:
derivativeKernels.d2_dlap = -2 * (
  1 / geometry.dx**2 * (1 - np.cos(geometry.KX * geometry.dx))
  + 1 / geometry.dy**2 * (1 - np.cos(geometry.KY * geometry.dy))
)
```

This model uses the substrate spectral `-(KX**2 + KY**2)`. The substitution is
deferred for a later refactor; when it returns it needs a home that does not put
a discretisation flag on a shared substrate kernel - a sibling module under
`discretization/`, not a branch in `spectral/kernels.py`.

### Deferred: the predictor-corrector step

The source fork ran a predictor-corrector pair. `GetExponentialTimesteppingKernels`
returned a third coefficient,

$$
\varphi_2(z) = \frac{e^{z} - (1 + z)}{\hat{L}^2}
$$

and a corrector stage applied it to the change in the nonlinear term, following
Cross, Meiron & Tu eq. 20:

```python
N_n_hat         = Gamma * kernel_laplacian * nonlin_mu_hat      + kernel_laplacian * externalV_hat
N_predicted_hat = Gamma * kernel_laplacian * nonlin_mu_next_hat + kernel_laplacian * externalV_hat
N_1_hat         = N_predicted_hat - N_n_hat
phi_next_hat[...] = phi_next_hat + kernel_nonlinexp2_phidt * N_1_hat
phi_next_hat.flat[0] = phi_hat.flat[0]  # enforce mass conservation
```

This model is ETD1: the predictor stage alone, which already carries the
external-potential force, so the reduction lowers the time order and changes
nothing structurally. Removing it meant dropping `calc_nonlin_mu_next_impl`,
`calc_final_phi_next_impl`, `_Batch4_CalcNonlinMuNext`,
`_Batch3_CalcFinalPhiNext`, the `kernel_nonlinexp2_phidt`, `batch4` and
`nonlin_mu_next` state allocations, and two parameters of
`Simulation.__init__`.

**The mass-conservation pin was relocated, not dropped.** The
`phi_next_hat.flat[0] = phi_hat.flat[0]` line above was the last statement of
the corrector; it now closes `calc_phi_next_impl`. At `k = 0` the kernel
Laplacian is zero so the pin is redundant in exact arithmetic, but it guards
against float drift and was part of the original time loop.

**One bug in the deferred code, for whoever restores it.** The corrector's
coefficient line read

```python
kNonlinearExp2Dt[~small_z] = (kLinearExpDt[~~small_z] - (1.0 + z[~small_z])) / (kLinear[~small_z] ** 2)
```

`~~small_z` is `small_z`. The left side is indexed by the large-`z` modes while
the first right-hand term reads the small-`z` ones. `kLinear_phi` vanishes only
at `k = 0`, so exactly one element is selected and NumPy broadcasts it rather
than raising: every large-`z` mode received `exp(z) = 1`. Reproduced on an 8x8
grid, the coefficient was wrong on all 63 non-zero modes with a maximum relative
error of 2.5e+01. The correct index is `kLinearExpDt[~small_z]`. Because
`pc=True` was the default and the only call, every run of the source fork is
affected.

### The reported free energy excludes the external potential

`CalculateFreeEnergy` is identical to the base forks': it sums the gradient,
ideal-gas and polynomial terms only. Neither the external-potential coupling nor
the `-M*phi` vacancy-forcing term contributes to the number it returns, so the
reported free energy is not the functional this model's dynamics actually
minimise. Carried over unchanged; worth knowing when reading the examples.

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

Migrated from `log/gpu-spectral-extV-implementation` (2026-10-12), staged under
`sandbox/log-pfc-migration/staged/`, in reduced form. Besides the two deferrals
above: the same six definitions moved to the spectral substrate,
`_VacancyTracker` moved to `pfc_models.analysis.vacancy_tracker`,
`_externalPotentials.py` became the model-local `external_potentials.py`,
imports rewritten, `smallZTol` threaded through the parameter builder at this
fork's `1e-8`, and the mass-conservation pin relocated into the predictor. The
active log clamp is untouched.

The second-order variant that shared this directory
(`Simulation2ndOrder.py`, a second-order expansion of the log free energy)
is **not** migrated and stays in the sandbox.

Full per-file record: [migration-log.md](../../../../../migration-log.md).

Migration checklist:

- [ ] finite-difference Laplacian refactored back in
- [ ] predictor-corrector refactored back in, with the `~~small_z` fix
- [x] substrate dependencies available (`geometry_2d`, `kernels`, `backends`)
- [x] clamp decision recorded (no change applied this step)
- [x] initial conditions draw from an explicit Generator
- [ ] `test_regression.py` written and passing - **omitted by decision for this migration step**; behaviour is verified visually from the example notebooks
- [ ] `model.toml` `migrated = true`
- [ ] source directory recorded as superseded in the old repo

## Results that used this model

_Paper, run-id, tag._
