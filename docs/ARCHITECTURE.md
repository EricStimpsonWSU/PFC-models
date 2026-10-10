# How the substrate is organized, and why it should not fragment

Four axes are known to be coming, each of which would otherwise fork the
substrate:

| axis | today | planned |
|---|---|---|
| dimensionality | 2D | 1D, 3D |
| discretization | spectral | finite difference; possibly finite element |
| boundaries | fully periodic | semi-periodic, non-periodic |
| backend | CuPy, NumPy+PyFFTW | Mac GPU (Metal/MPS) |

A naive `core/` becomes unreadable under that load — every module grows flags
for cases it was not written for. The structure below avoids it without
introducing a class hierarchy.

## The rule

**Substrate modules are leaf modules with a documented contract.** Siblings
expose the same names and the same attribute semantics; they do not inherit from
each other and there is no abstract base.

```
src/pfc_models/
├── backends/                  one file per array/FFT backend
│   ├── numpy_fftw.py
│   ├── cupy_cuda.py
│   └── torch_mps.py           stub
├── discretization/
│   ├── spectral/              periodic by construction
│   │   ├── geometry_1d.py  geometry_2d.py  geometry_3d.py
│   │   └── kernels.py         derivative, Gaussian, ETD kernel builders
│   ├── finitediff/            boundaries are explicit here
│   │   ├── geometry_2d.py
│   │   ├── stencils.py
│   │   └── boundaries.py      periodic | semi-periodic | Dirichlet | Neumann
│   └── fem/                   placeholder
├── fields/                    initial conditions, per dimensionality
├── analysis/                  atom and vacancy tracking
├── io/                        run ids, config hashing, frame writing
└── viz/                       density heatmaps
```

Adding 3D spectral support means writing `spectral/geometry_3d.py` — not
editing `geometry_2d.py` to take a `ndim` argument. Adding non-periodic
boundaries means `finitediff/boundaries.py` grows a function, not
`spectral/kernels.py` growing a branch it can never satisfy.

## The contract

A `Geometry`, whatever module it comes from, exposes the same attribute names
(`shape`, `dx`, `X`, `Y`, …, and the index/coordinate conversions). A model
written against those names runs on any geometry whose contract it satisfies,
and fails loudly rather than silently when it does not. The contract is
documented in the module docstrings of `discretization/spectral/geometry_2d.py`
— the reference implementation — and every sibling repeats it.

## Why this stays honest

Sibling substrate modules are exactly the situation that produced the drift in
the original codebase: parallel files, a fix applied to one. So
`variant_check.py` is pointed at the sibling sets as well as at the models. A
change to `spectral/geometry_2d.py` that does not reach `geometry_3d.py` shows
up as `DIVERGENCE INCREASED` at commit time, and is either propagated or
explicitly accepted.
