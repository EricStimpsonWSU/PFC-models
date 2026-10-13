"""Spectral operator kernels: derivatives, local (Gaussian) averaging, ETD timestepping.

Provenance and the three naming/reconciliation decisions
--------------------------------------------------------
`_DerivativeKernels` and `BuildDerivativeKernels` come from
log/gpu-spectral-base-implementation/Simulation.py. The base forks agree; the
extV fork had replaced the spectral Laplacian with the Fourier symbol of the
five-point finite-difference Laplacian, and that substitution is deferred, so
`d2_dlap` here is the spectral `-(KX**2 + KY**2)`.

`Get_kGaussian` is migrated as `kernel_local`, not as the `kernel_gaussian`
the source-tree audit suggested: the model code already binds `kernel_gaussian`
as a local name for the kernel *array*
(`kernel_gaussian = self.State.kernel_gaussian`), so a function of that name
would be shadowed in three methods per model. `kernel_local` names what the
kernel is for - it low-pass filters phi and mu into the `local_phi_hat` and
`local_mu_hat` fields - and collides with nothing.

`kernel_etd` replaces `GetExponentialTimesteppingKernels`, which had four bodies
across the source tree. The two base forks differed only in an `np.errstate`
wrapper, which is kept here and is inert under CuPy; the hard-coded `2e-5`
threshold is now the `small_z_tol` keyword, defaulting to the value both base
models were run at. `np.exp` and `np.where` dispatch to CuPy through
`__array_ufunc__`, which the GPU forks already rely on, so one copy serves both
backends. The predictor-corrector third coefficient is deferred and is not
computed here.
"""

# numerics
import numpy as np


class _DerivativeKernels:
  def __init__(self):
    return None


def BuildDerivativeKernels(geometry):
  derivativeKernels = _DerivativeKernels()
  derivativeKernels.Geometry = geometry
  derivativeKernels.d_dx = 1j * geometry.KX
  derivativeKernels.d_dy = 1j * geometry.KY
  derivativeKernels.d2_dx2 = -geometry.KX**2
  derivativeKernels.d2_dy2 = -geometry.KY**2
  derivativeKernels.d2_dlap = -(geometry.KX**2 + geometry.KY**2)
  return derivativeKernels


def kernel_local(geometry, w):
  # Create a verifiable kGaussian field
  kGaussian = np.exp(-((geometry.KX)**2 + (geometry.KY)**2)*w**2/2)
  return kGaussian


def kernel_etd(kLinear, dt, *, small_z_tol=2e-5):
  """
  ETD1 timestepping kernels.

  small_z_tol guards the removable singularity in (e^z - 1)/kLinear at
  z = kLinear*dt -> 0; below it the Taylor expansion is used instead.
  """
  z = kLinear * dt
  kLinearExpDt = np.exp(z)
  with np.errstate(divide='ignore', invalid='ignore'):
    kNonlinearExpDt = np.where(
      np.abs(z) > small_z_tol,
      (kLinearExpDt - 1) / kLinear,
      dt * (1 + z / 2 + z**2 / 6)
      )

  return kLinearExpDt, kNonlinearExpDt
