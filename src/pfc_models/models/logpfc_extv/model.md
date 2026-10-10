# log-PFC with external potential (vacancy forcing)

**Status:** not migrated · **Declared features:** see [model.toml](model.toml)

## Free energy

_Write the functional. State every symbol._

## Equations of motion

_The evolution equations as implemented, including the time-stepping scheme._

## Assumptions

_Dimensionality, boundary conditions, what is held fixed, what the model cannot
represent._

## Numerical notes

### Clamping of log(phi)

**Decided:** the logarithm alone is clamped —

```python
nonlin_mu = where(phi > minPhi, log(phi), minLog) + g * phi**2 + phi**3
```

Clamping the full expression (`where(phi > minPhi, log(phi) + g*phi**2 + phi**3,
minLog)`) is an implementation bug: it discards the polynomial terms wherever
`phi <= minPhi`. Four of the nine original forks carried the buggy form.

Adds the `M` term to the linear mu kernel and the external-potential machinery. A second-order variant (Simulation2ndOrder.py) also exists and may warrant its own catalogue entry.

## Provenance

Migrated from `log/gpu-spectral-extV-implementation`.

Migration checklist:

- [ ] substrate dependencies available (`geometry`, `kernels`, `backends`)
- [ ] clamp expression correct per the section above
- [ ] initial conditions take an explicit `rng`, not a global seed
- [ ] `test_regression.py` written and passing
- [ ] `model.toml` `migrated = true`
- [ ] source directory recorded as superseded in the old repo

## Results that used this model

_Paper, run-id, tag._
