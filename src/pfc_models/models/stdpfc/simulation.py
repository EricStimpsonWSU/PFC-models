"""standard PFC, 2D spectral, GPU (CuPy).

Self-contained implementation. Duplication with sibling models is deliberate -
see ../../../../docs/ARCHITECTURE.md.

Provenance: authored in-repo from models/logpfc as the structural template. The
free energy and chemical potential are the standard-PFC branch of
papers/2026-log-pfc/derivations/c1-positive-density/01-model-parameter-conversion.md
(section 2, canonical form (S)). Differences from logpfc, all of them physics or
naming, none structural:

  * the density field is `n`, not `phi` - the standard branch Taylor-expands the
    entropic term, so there is no positivity constraint and no `ln` to clamp;
  * `nonlin_mu` is `g*n**2 + n**3`: the `ln(phi)` term is gone, so `minPhi` and
    `minLog` are not parameters of this model;
  * the free-energy density carries `(epsilon/2)*n**2 + (g/3)*n**3 + n**4/4`
    with no entropic contribution.

The cubic coefficient `g` is retained rather than removed by the field shift
n -> n + g/3 (derivation sections 2.2 and 2.3), so that the removability can be
verified by running this model against its shifted equivalent. See model.md.
"""
from collections.abc import Callable

# numerics
import numpy as np
import cupy as cp

from pfc_models.backends.cupy_cuda import npFloatType, cpFloatType, cpComplexType

# substrate
from pfc_models.discretization.spectral.geometry_2d import Geometry, Build2DGeometry
from pfc_models.discretization.spectral.kernels import (
  _DerivativeKernels,
  BuildDerivativeKernels,
  kernel_local,
  kernel_etd,
)

class ModelParameters:
  """
  Class that holds model parameters for the simulation. Datatypes are automatically converted to
  those defined by the backend. For example, if a parameter is set to any numeric type, it will
  be converted to npFloatType.
  """
  def __init__(self):
    return None

  def __setattr__(self, name, value):
    # convert scalars to npFloatType
    if isinstance(value, (int, npFloatType)):
      value = npFloatType(value)

    # convert lists/tuples to npFloatType arrays
    elif isinstance(value, (list, tuple)):
      value = np.array(value, dtype=npFloatType)

    super().__setattr__(name, value)

def BuildStdPFCModelParameters(
    Gamma = 1.0,
    beta = 1.0,
    q0 = 1.0,
    epsilon = 1.0,
    g = 1.0,
    dt = 0.01,
    meanDensity = None,
    smallZTol = 2e-5
):
  model = ModelParameters()
  model.Gamma = Gamma
  model.beta = beta
  model.q0 = q0
  model.epsilon = epsilon
  model.g = g
  model.dt = dt
  model.meanDensity = meanDensity
  model.smallZTol = smallZTol

  return model


def GetLinearKernel_mu(modelParameters, DerivativeKernels):
  beta = modelParameters.beta
  epsilon = modelParameters.epsilon
  q0 = modelParameters.q0

  kernel_lap = DerivativeKernels.d2_dlap
  kLinear_mu = beta * (q0**2 + kernel_lap)**2 + epsilon
  return kLinear_mu

def GetLinearKernel_n(modelParameters, DerivativeKernels):
  Gamma = modelParameters.Gamma
  kernel_lap = DerivativeKernels.d2_dlap
  kLinear_n = Gamma * kernel_lap * GetLinearKernel_mu(modelParameters, DerivativeKernels)
  return kLinear_n


class _SimulationState:
  def __init__(self, modelParameters, derivativeKernels):
    # simulation time
    self.t = npFloatType(0)

    # kSpace, rSpace sizes
    self.kSpaceShape = derivativeKernels.Geometry.KX.shape
    self.rSpaceShape = derivativeKernels.Geometry.X.shape

    # derivative kernels
    self.kernel_grad = cp.asarray(np.stack((derivativeKernels.d_dx, derivativeKernels.d_dy), axis=0), dtype=cpComplexType)
    self.kernel_d_dx = self.kernel_grad[0]  # view
    self.kernel_d_dy = self.kernel_grad[1]  # view
    self.kernel_laplacian = cp.asarray(derivativeKernels.d2_dlap.real, dtype=cpFloatType)

    # gaussian kernel
    self.kernel_gaussian = cp.zeros(self.kSpaceShape, dtype=cpFloatType)
    self.kernel_gaussian[...] = cp.asarray(kernel_local(derivativeKernels.Geometry, w=2*np.pi), dtype=cpFloatType)

    # exponential timestepping kernels [precomputed for efficiency or inline for memory efficiency]
    self.kernel_lin_mu = cp.asarray(GetLinearKernel_mu(modelParameters, derivativeKernels), dtype=cpFloatType)
    dt = modelParameters.dt

    kernel_lin_n = cp.asarray(GetLinearKernel_n(modelParameters, derivativeKernels), dtype=cpFloatType)
    kernel_linexp_ndt, kernel_nonlinexp_ndt = kernel_etd(kernel_lin_n, dt, small_z_tol=modelParameters.smallZTol)
    self.kernel_linexp_ndt = cp.asarray(kernel_linexp_ndt, dtype=cpFloatType)    # precomputed
    self.kernel_nonlinexp_ndt = cp.asarray(kernel_nonlinexp_ndt, dtype=cpFloatType) # precomputed

    # buffer blocks
    self.buffer_rSpace = cp.zeros((33, *self.rSpaceShape), dtype=cpFloatType)
    self.buffer_kSpace = cp.zeros((33, *self.kSpaceShape), dtype=cpComplexType)

    # batch1 FFT (FWD):
    self.batch1 = self.buffer_rSpace[0:2]
    self.batch1_hat = self.buffer_kSpace[0:2]

    # named fields in batch1:
    self.n = self.batch1[0]
    self.n_hat = self.batch1_hat[0]

    self.nonlin_mu = self.batch1[1]
    self.nonlin_mu_hat = self.batch1_hat[1]
    # end of batch1, total length: 2

    # batch2 FFT (INV):
    self.batch2 = self.buffer_rSpace[2:4]
    self.batch2_hat = self.buffer_kSpace[2:4]

    # named fields in batch2:
    # μ
    self.mu = self.batch2[0]
    self.mu_hat = self.batch2_hat[0]

    # ⟨μ⟩
    self.local_mu = self.batch2[1]
    self.local_mu_hat = self.batch2_hat[1]
    # end of batch2, total length: 2

    # batch3 FFT (INV):
    self.batch3 = self.buffer_rSpace[4:5]
    self.batch3_hat = self.buffer_kSpace[4:5]

    # named fields in batch3
    self.n_next = self.batch3[0]
    self.n_next_hat = self.batch3_hat[0]
    # end of batch3, total length: 1

class Simulation:
  def __init__(
      self,
      calcNonLinMu_impl: Callable[['Simulation'], None] | None = None,
      calcMu_Terms_kSpace_impl: Callable[['Simulation'], None] | None = None,
      calcNNext_impl: Callable[['Simulation'], None] | None = None,):

    # Primary members
    self.Model = None
    self.Geometry = None
    self.State = None

    # Convenience views
    self.n = None
    self.mu = None

    # Assign method implementations
    if calcNonLinMu_impl is None:
      raise TypeError("calcNonLinMu_impl must be a callable")
    self._CalcNonlinMu_impl = calcNonLinMu_impl

    if calcMu_Terms_kSpace_impl is None:
      raise TypeError("calcMu_Terms_kSpace_impl must be a callable")
    self._CalcMu_Terms_kSpace_impl = calcMu_Terms_kSpace_impl

    if calcNNext_impl is None:
      raise TypeError("calcNNext_impl must be a callable")
    self._CalcNNext_impl = calcNNext_impl

  @property
  def t(self):
    """
    Convenience alias for simulation time.
    """
    return self.State.t

  # Element-wise kernels (internal)
  def _CalcNonlinMu(self):
    self._CalcNonlinMu_impl(self)

  def _CalcMu_Terms_kSpace(self):
    self._CalcMu_Terms_kSpace_impl(self)

  def _CalcNNext(self):
    self._CalcNNext_impl(self)

  # Batches (kernel followed by FFT, internal)
  def _Batch1_CalcNonlinMu(self):
    self._CalcNonlinMu()
    self.State.batch1_hat[...] = cp.fft.rfft2(self.State.batch1, axes=(-2,-1))

  def _Batch2_CalcMu_Terms(self):
    self._CalcMu_Terms_kSpace()
    self.State.batch2[...] = cp.fft.irfft2(self.State.batch2_hat, s=self.State.batch2.shape[-2:], axes=(-2,-1))

  def _Batch3_CalcNNext(self):
    self._CalcNNext()
    self.State.batch3[...] = cp.fft.irfft2(self.State.batch3_hat, s=self.State.batch3.shape[-2:], axes=(-2, -1))

  def TimeStep(self):

    # Run FFT batches sequentially
    self._Batch1_CalcNonlinMu()
    self._Batch2_CalcMu_Terms()
    self._Batch3_CalcNNext()

    # n(t) ⟵ n(t + Δt)
    self.State.n[...] = self.State.n_next

    self.State.t += self.Model.dt

  def Get_local_n(self):
    """
    Returns a coarsened density field.
    """
    kernel_gaussian = self.State.kernel_gaussian
    n_hat = self.State.n_hat

    local_n_hat = kernel_gaussian * n_hat
    local_n = cp.fft.irfft2(local_n_hat, s=self.State.n.shape, axes=(-2,-1))

    return local_n.get()

  def CalculateFreeEnergy(self):
    """
    Calculates the free energy of the system.
    """
    beta = self.Model.beta
    epsilon = self.Model.epsilon
    g = self.Model.g
    q0 = self.Model.q0
    n = self.State.n

    n_hat = cp.fft.rfft2(n, axes=(-2,-1))
    kernel_laplacian = self.State.kernel_laplacian
    kernel_lin_f = (beta / 2.0) * (q0**2 + kernel_laplacian)**2
    lin_f_hat = kernel_lin_f * n_hat
    lin_f = cp.fft.irfft2(lin_f_hat, s=n.shape, axes=(-2,-1))
    nonlin_f = (epsilon / 2.0) * n**2 + (g / 3.0) * n**3 + (1 / 4.0) * n**4

    f = n * lin_f + nonlin_f

    return f.get().mean()  # Return as a NumPy scalar

def BuildCupyGPU_Simulation(modelParameters, geometry):
  simulation = Simulation(
    calc_nonlin_mu_impl,
    calc_mu_terms_kspace_impl,
    calc_n_next_impl)
  simulation.Model = modelParameters
  simulation.Geometry = geometry

  # construct derivative kernels
  derivativeKernels = BuildDerivativeKernels(geometry)
  simulation.State = _SimulationState(modelParameters, derivativeKernels)

  # views of primary fields for user convenience
  simulation.n = simulation.State.n
  simulation.mu = simulation.State.mu

  return simulation

def calc_nonlin_mu_impl(self: Simulation):
  """
  Calculates the nonlinear term of the chemical potential μ in real space.
  """
  n = self.State.n
  nonlin_mu = self.State.nonlin_mu
  g = self.Model.g

  # compute nonlinear chemical potential μ = gn² + n³
  nonlin_mu[...] = g * n**2 + n**3

def calc_mu_terms_kspace_impl(self: Simulation):
  """
  Calculates μ, ⟨μ⟩ in k-space.
  """
  # Calc μ (k-space):
  kernel_lin_mu = self.State.kernel_lin_mu
  n_hat = self.State.n_hat
  nonlin_mu_hat = self.State.nonlin_mu_hat
  mu_hat = self.State.mu_hat
  mu_hat[...] = kernel_lin_mu * n_hat + nonlin_mu_hat

  # Calc ⟨μ⟩ (k-space):
  kernel_gaussian = self.State.kernel_gaussian
  local_mu_hat = self.State.local_mu_hat
  local_mu_hat[...] = kernel_gaussian * mu_hat

def calc_n_next_impl(self: Simulation):
  """
  Calculates the next time step of the density field n in k-space.
  """
  Gamma = self.Model.Gamma
  kernel_linexp_ndt = self.State.kernel_linexp_ndt
  kernel_nonlinexp_ndt = self.State.kernel_nonlinexp_ndt
  kernel_laplacian = self.State.kernel_laplacian
  n_hat = self.State.n_hat
  nonlin_mu_hat = self.State.nonlin_mu_hat
  n_next_hat = self.State.n_next_hat
  n_next_hat[...] = (
    kernel_linexp_ndt * n_hat +
    kernel_nonlinexp_ndt * (
      Gamma * kernel_laplacian * nonlin_mu_hat
    )
  )
