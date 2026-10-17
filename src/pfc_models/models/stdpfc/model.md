# standard PFC, 2D spectral, GPU

**Status:** staged for review (2026-10-16) · **Declared features:** see [model.toml](model.toml)

## Free energy

The free-energy *density* as implemented in `Simulation.CalculateFreeEnergy`:

$$
f[n] \;=\; n\,\frac{\beta}{2}\bigl(q_0^2 + \nabla^2\bigr)^2 n
\;+\; \frac{\varepsilon}{2}n^2
\;+\; \frac{g}{3}n^3 \;+\; \frac{1}{4}n^4
$$

`CalculateFreeEnergy` returns the mean of $f$ over the box.

| symbol | code | meaning |
|---|---|---|
| $n$ | `State.n` | dimensionless density deviation from the reference liquid |
| $\beta$ | `Model.beta` | stiffness of the gradient (peak-selection) term |
| $q_0$ | `Model.q0` | reference wavenumber selecting the lattice spacing |
| $\varepsilon$ | `Model.epsilon` | quadratic coefficient (temperature-like) |
| $g$ | `Model.g` | cubic coefficient |
| $\Gamma$ | `Model.Gamma` | mobility in the equation of motion |

This is the canonical standard-PFC form (S) of the parameter-conversion
derivation: the branch that Taylor-expands the entropic term $(1+n)\ln(1+n)$
about $n = 0$ and truncates at $n^4$, absorbing the expansion coefficients into
the polynomial coefficients. The log-PFC family keeps that term exactly and
works in $\phi = 1 + n$ instead. Consequences of the difference that matter when
reading this code next to `logpfc`:

* the field is $n$, not $\phi$, and **$n$ is not bounded below** - positivity is
  not enforced and is not meaningful here. There is no logarithm, so there is no
  `minPhi`/`minLog` clamp and those are not parameters of this model;
* $n = 0$ is the reference liquid, where $\phi = 1$ is;
* the two models' parameters are *not* interchangeable. The dictionary is in the
  derivation, sections 5.1-5.3; at $\alpha = 1$,
  $\beta_s = (3/4)^{1/2}\beta_\ell$,
  $\varepsilon_s = (3/4)^{1/2}(\varepsilon_\ell + 2g_\ell + 4)$,
  $g_s = (3/4)^{3/4}(g_\ell + 5/2)$, $n = (3/4)^{1/4}(\phi - 1)$.

## Equations of motion

Conserved (Cahn-Hilliard) dynamics driven by the chemical potential:

$$
\frac{\partial n}{\partial t} \;=\; \Gamma \nabla^2 \mu,
\qquad
\mu \;=\; g n^2 + n^3
      + \Bigl[\beta\bigl(q_0^2 + \nabla^2\bigr)^2 + \varepsilon\Bigr]n
$$

Split for time stepping into a linear operator and a nonlinear remainder,

$$
\hat{L} = \Gamma k^2_{\text{lap}} \Bigl[\beta\bigl(q_0^2 + k^2_{\text{lap}}\bigr)^2
          + \varepsilon\Bigr],
\qquad
\hat{N} = \Gamma k^2_{\text{lap}} \,\widehat{\bigl(g n^2 + n^3\bigr)}
$$

with $k^2_{\text{lap}} = -(k_x^2 + k_y^2)$, and advanced by first-order
exponential time differencing (ETD1):

$$
\hat n(t + \Delta t) \;=\; e^{\hat{L}\Delta t}\,\hat n(t)
   \;+\; \varphi_1(\hat{L}\Delta t)\,\hat{N}(t),
\qquad
\varphi_1(z) = \frac{e^{z} - 1}{\hat{L}}
$$

`kernel_etd` in the spectral substrate returns the two coefficients. Near
$z = \hat{L}\Delta t \to 0$ the quotient is a removable singularity, so below
`small_z_tol` it uses the Taylor form $\Delta t\,(1 + z/2 + z^2/6)$.

## Assumptions

* Two dimensions, fully periodic, uniform rectangular grid.
* Spectral spatial discretisation: all derivatives are Fourier multipliers, so
  the solution is assumed smooth and band-limited on the grid.
* `float64` throughout.
* Mass is conserved: the dynamics are of divergence form and the `k = 0` mode of
  the update carries no nonlinear contribution.
* The quartic truncation is assumed valid, i.e. the run stays in a range of $n$
  where the expansion of the entropic term is a good approximation. Nothing in
  the code enforces this, and nothing stops $n$ from passing below the image of
  $\rho = 0$.
* The model cannot represent free surfaces, non-periodic boundaries, or any
  dimensionality other than 2 - those are separate catalogue entries by design.

## Numerical notes

### No clamp

The log-PFC family clamps the logarithm of the density
(`where(phi > minPhi, log(phi), minLog) + ...`). This model has no logarithm and
therefore no clamp, no `minPhi` and no `minLog`. A reader arriving from
`logpfc` should read the absence as deliberate, not as an omission.

### The cubic coefficient is retained

$g$ is removable in the standard model: substituting $n \to \tilde n = n + g/3$
leaves a quartic with no cubic term, quadratic coefficient
$\varepsilon_0 = \varepsilon - g^2/3$, and the generated linear and constant
terms are annihilated by the $\nabla^2$ in front of $\mu$ (derivation sections
2.2-2.3). The two models therefore generate *identical dynamics*, not merely
similar free energies.

This model keeps $g$ anyway, so that the removability can be checked by running
rather than assumed. The paired run is:

| | run A | run B |
|---|---|---|
| parameters | $(\beta,\ \varepsilon,\ g)$ | $(\beta,\ \varepsilon - g^2/3,\ 0)$ |
| initial condition | $n_0$ | $n_0 + g/3$ |
| mean density | $\bar n$ | $\bar n + g/3$ |

Expected: $n_B(t) = n_A(t) + g/3$ pointwise at every step, to time-stepper
precision. The free energies are **not** equal - run B's density differs from
run A's by terms linear and constant in $n$, whose box average is fixed because
the mean is conserved. So the right check on the energy is that the difference
is constant in time, with the value

$$
\langle f_B\rangle - \langle f_A\rangle
= \beta q_0^4 s\Bigl(\bar n + \tfrac{s}{2}\Bigr)
+ \bigl(\varepsilon s - 2s^3\bigr)\bar n
+ \tfrac12 \varepsilon s^2 - \tfrac54 s^4,
\qquad s = \frac{g}{3},
$$

$\bar n$ being run A's mean density. Comparing $\langle f\rangle$ between the
two runs without this offset will look like a discrepancy and is not one.

Both identities were checked numerically against the substrate kernels on a
64×64 box at $(\beta,\varepsilon,g) = (0.433013, -0.779423, 0.402964)$: the
energy offset reproduces the closed form to $3\times10^{-18}$, and
$\mu_B - \mu_A$ is constant across the box to $7\times10^{-13}$ with value
$\varepsilon s - 2s^3 + \beta q_0^4 s$, which $\nabla^2$ annihilates.


Note also that the removal moves the density origin: $\rho = \rho_\ell$ sits at
$n = 0$ in run A but at $\tilde n = g/3$ in run B. Any phase diagram plotted
against the mean density has to say which convention it is in.

### The small-z threshold

The ETD coefficients come from the substrate function `kernel_etd`, whose guard
against the removable singularity in `(exp(z) - 1)/kLinear` is the keyword
`small_z_tol` rather than a hard-coded constant, settable per run through
`BuildStdPFCModelParameters(..., smallZTol=...)`. This model declares
**`smallZTol = 2e-5`**, matching the log-PFC family.

### Initial-condition seeding

The generators in `pfc_models.fields.initial_conditions_2d` that draw randomness
(`GenerateHotInitialConditions`, `GenerateLogNormalHotInitialConditions`) take
their stream from `np.random.default_rng(seed)`. Both of them force a positive
mean density, which suits the log family; for this model the perfect-crystal
generators, which are additive about `meanDensity`, are the natural choice and
accept a negative mean.

## Tests

| file | what it covers |
|---|---|
| [../../../../tests/test_smoke_models.py](../../../../tests/test_smoke_models.py) | numeric stability: free-energy relaxation and mass conservation, one test per model |
| [test_regression.py](test_regression.py) | free energy at a pinned configuration and step against a recorded reference; plus this model's closed-form checks - single-mode free energy, ETD1 exactness in the linear regime, both cubic-removability statements |

Run both runtimes with `tools/run-tests.ps1`; every test gates on the backend
the active runtime can import, so each venv covers one variant and reports the
other as skipped.

## Provenance

Authored in-repo (2026-10-16), not migrated: `models/logpfc` supplied the
structure and `papers/2026-log-pfc/derivations/c1-positive-density/01-model-parameter-conversion.md`
(section 2, canonical form (S)) supplied the physics. Relative to the template
the differences are the density field name (`n` for `phi`), the removal of the
entropic term from both the chemical potential and the free energy, and the
removal of `minPhi`/`minLog` from the parameter set. Nothing structural changed:
the state layout, the FFT batching, the ETD split and the builder signatures are
the template's.

Checklist:

- [x] substrate dependencies available (`geometry_2d`, `kernels`, `backends`)
- [x] free energy and chemical potential checked against derivation section 2
- [x] example notebook executed (2026-10-16)
- [ ] **test suite not yet run on a CUDA device** - the authoring session had no
      usable one (CuPy reaches NVRTC and the compile aborts the process), so the
      CuPy half of the suite is written but unexecuted. Run
      `tools/run-tests.ps1 -SkipCpu`.
- [ ] `test_regression.py` reference recorded - `tools/run-tests.ps1 -RecordGpuReference`
- [x] `tools/build_feature_matrix.py` re-run and `tools/variant_check.py accept`
      re-baselined

## Results that used this model

_Paper, run-id, tag._
