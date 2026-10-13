"""CuPy / CUDA (NVIDIA GPU) backend.

Sibling of the other modules in `pfc_models.backends`. Carries the host and
device dtype aliases; the FFT half of the backend contract
(`fftn`/`rfftn`/`to_numpy`/`xp`) is deliberately absent, because the migrated
models call `cp.fft` directly in `_SimulationState` exactly as the source forks
did. Providing a dispatch layer the models do not use would be new code, not a
relocation.

Provenance: log/gpu-spectral-base-implementation/DataTypes.py (byte-identical to
the gpu-spectral-extV copy).
"""

# numerics
import numpy as np
import cupy as cp

# numerical types used in simulation
npFloatType = np.float64
npComplexType = np.complex128
cpFloatType = cp.float64
cpComplexType = cp.complex128

name = "cupy_cuda"


def available() -> bool:
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False
