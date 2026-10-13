"""NumPy + PyFFTW (CPU) backend.

Sibling of the other modules in `pfc_models.backends`. Carries the host dtype
aliases; the FFT half of the backend contract (`fftn`/`rfftn`/`to_numpy`/`xp`)
is deliberately absent, because the migrated models plan and call their
transforms directly in `_SimulationState` exactly as the source forks did.
Providing a dispatch layer the models do not use would be new code, not a
relocation.

Provenance: log/cpu-spectral-base-implementation/DataTypes.py
"""

# numerics
import numpy as np

# numerical types used in simulation
npFloatType = np.float64
npComplexType = np.complex128

name = "numpy_fftw"


def available() -> bool:
    try:
        import pyfftw  # noqa: F401
    except ImportError:
        return False
    return True
