# log-PFC, 2D spectral, GPU

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

The audit found this fork applies **no clamp** to log(phi). The migrated version must clamp the logarithm only.

## Provenance

Migrated from `log/gpu-spectral-base-implementation`.

Migration checklist:

- [ ] substrate dependencies available (`geometry`, `kernels`, `backends`)
- [ ] clamp expression correct per the section above
- [ ] initial conditions take an explicit `rng`, not a global seed
- [ ] `test_regression.py` written and passing
- [ ] `model.toml` `migrated = true`
- [ ] source directory recorded as superseded in the old repo

## Results that used this model

_Paper, run-id, tag._
