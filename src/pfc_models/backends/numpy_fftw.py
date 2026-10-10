"""NumPy + PyFFTW (CPU) backend.

Sibling of the other modules in `pfc_models.backends`. Every backend module
exposes the same names so a model can select one at construction time:

    xp          array module (numpy, cupy, torch, ...)
    fftn/ifftn  forward/inverse n-D FFT on real input
    rfftn/irfftn
    to_numpy(a) host copy as a numpy array
    available() -> bool
    name        str

Nothing here is a base class and nothing imports another backend. Add a new
backend by adding a sibling file that satisfies the same names.

STUB — not implemented. Migrate from the CPU fork's pyfftw plan setup.
"""

name = "numpy_fftw"


def available() -> bool:
    raise NotImplementedError("numpy_fftw backend not implemented yet")
