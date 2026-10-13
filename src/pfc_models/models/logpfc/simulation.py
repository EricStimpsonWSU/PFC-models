"""log-PFC, 2D spectral, GPU (CuPy).

Self-contained implementation. Duplication with sibling models is deliberate -
see ../../../../docs/ARCHITECTURE.md.

Provenance: log/gpu-spectral-base-implementation/Simulation.py. Relocated with
Geometry, Build2DGeometry, _DerivativeKernels, BuildDerivativeKernels,
Get_kGaussian and GetExponentialTimesteppingKernels
moved to the spectral substrate, imports rewritten, the Gaussian-kernel and
ETD call sites renamed, and `smallZTol` threaded through the parameter builder.
No physics changed; the commented-out log clamp is carried verbatim. See
model.md and ../../../../../migration-log.md.
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
  those defined in DataTypes.py. For example, if a parameter is set any numeric type, it will
  be converted to npFloatType.
  """
  def __init__(self):
    return None
    
  def __setattr__(self, name, value):
    # convert scalars to float32
    if isinstance(value, (int, npFloatType)):
      value = npFloatType(value)

    # convert lists/tuples to float32 arrays
    elif isinstance(value, (list, tuple)):
      value = np.array(value, dtype=npFloatType)

    super().__setattr__(name, value)

def BuildLogPFCModelParameters(
    Gamma = 1.0,
    beta = 1.0,
    q0 = 1.0,
    epsilon = 1.0,
    g = 1.0,
    dt = 0.01,
    minPhi = 1e-6,
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
  model.minPhi = minPhi
  model.meanDensity = meanDensity
  model.smallZTol = smallZTol

  # calculate log of minPhi
  model.minLog = np.log(minPhi)
  return model


def GetLinearKernel_mu(modelParameters, DerivativeKernels):
  beta = modelParameters.beta
  epsilon = modelParameters.epsilon
  q0 = modelParameters.q0

  kernel_lap = DerivativeKernels.d2_dlap
  kLinear_mu = beta * (q0**2 + kernel_lap)**2 + epsilon
  return kLinear_mu

def GetLinearKernel_phi(modelParameters, DerivativeKernels):
  Gamma = modelParameters.Gamma
  kernel_lap = DerivativeKernels.d2_dlap
  kLinear_phi = Gamma * kernel_lap * GetLinearKernel_mu(modelParameters, DerivativeKernels)
  return kLinear_phi


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

    kernel_lin_phi = cp.asarray(GetLinearKernel_phi(modelParameters, derivativeKernels), dtype=cpFloatType)
    kernel_linexp_phidt, kernel_nonlinexp_phidt = kernel_etd(kernel_lin_phi, dt, small_z_tol=modelParameters.smallZTol)
    self.kernel_linexp_phidt = cp.asarray(kernel_linexp_phidt, dtype=cpFloatType)    # precomputed
    self.kernel_nonlinexp_phidt = cp.asarray(kernel_nonlinexp_phidt, dtype=cpFloatType) # precomputed

    # buffer blocks
    self.buffer_rSpace = cp.zeros((33, *self.rSpaceShape), dtype=cpFloatType)
    self.buffer_kSpace = cp.zeros((33, *self.kSpaceShape), dtype=cpComplexType)

    # batch1 FFT (FWD):
    self.batch1 = self.buffer_rSpace[0:2]
    self.batch1_hat = self.buffer_kSpace[0:2]

    # named fields in batch1:
    self.phi = self.batch1[0]
    self.phi_hat = self.batch1_hat[0]

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

    # named fields in batch8
    self.phi_next = self.batch3[0]
    self.phi_next_hat = self.batch3_hat[0]
    # end of batch3, total length: 1

class Simulation:
  def __init__(
      self,
      calcNonLinMu_impl: Callable[['Simulation'], None] | None = None,
      calcMu_Terms_kSpace_impl: Callable[['Simulation'], None] | None = None,
      calcPhiNext_impl: Callable[['Simulation'], None] | None = None,):

    # Primary members
    self.Model = None
    self.Geometry = None
    self.State = None

    # Convience views
    self.phi = None
    self.mu = None

    # Assign ethod implementations
    if calcNonLinMu_impl is None:
      raise TypeError("calcNonLinMu_impl must be a callable")
    self._CalcNonlinMu_impl = calcNonLinMu_impl

    if calcMu_Terms_kSpace_impl is None:
      raise TypeError("calcMu_Terms_kSpace_impl must be a callable")
    self._CalcMu_Terms_kSpace_impl = calcMu_Terms_kSpace_impl

    if calcPhiNext_impl is None:
      raise TypeError("calcPhiNext_impl must be a callable")
    self._CalcPhiNext_impl = calcPhiNext_impl

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

  def _CalcPhiNext(self):
    self._CalcPhiNext_impl(self)

  # Batches (kernel followed by FFT, internal)
  def _Batch1_CalcNonlinMu(self):
    self._CalcNonlinMu()
    self.State.batch1_hat[...] = cp.fft.rfft2(self.State.batch1, axes=(-2,-1))
  
  def _Batch2_CalcMu_Terms(self):
    self._CalcMu_Terms_kSpace()
    self.State.batch2[...] = cp.fft.irfft2(self.State.batch2_hat, s=self.State.batch2.shape[-2:], axes=(-2,-1))

  def _Batch3_CalcPhiNext(self):
    self._CalcPhiNext()
    self.State.batch3[...] = cp.fft.irfft2(self.State.batch3_hat, s=self.State.batch3.shape[-2:], axes=(-2, -1))

  def TimeStep(self):

    # Run FFT batches sequentially
    self._Batch1_CalcNonlinMu()
    self._Batch2_CalcMu_Terms()
    self._Batch3_CalcPhiNext()

    # ϕ(t) ⟵ ϕ(t + Δt)
    self.State.phi[...] = self.State.phi_next

    self.State.t += self.Model.dt

  def Get_local_phi(self):
    """
    Returns a coarsened density field.
    """
    kernel_gaussian = self.State.kernel_gaussian
    phi_hat = self.State.phi_hat

    local_phi_hat = kernel_gaussian * phi_hat
    local_phi = cp.fft.irfft2(local_phi_hat, s=self.State.phi.shape, axes=(-2,1))

    return local_phi.get()

  def CalculateFreeEnergy(self):
    """
    Calculates the free energy of the system.
    """
    beta = self.Model.beta
    epsilon = self.Model.epsilon
    g = self.Model.g
    q0 = self.Model.q0
    phi = self.State.phi

    phi_hat = cp.fft.rfft2(phi, axes=(-2,-1))
    kernel_laplacian = self.State.kernel_laplacian
    kernel_lin_f = (beta / 2.0) * (q0**2 + kernel_laplacian)**2
    lin_f_hat = kernel_lin_f * phi_hat
    lin_f = cp.fft.irfft2(lin_f_hat, s=phi.shape, axes=(-2,-1))
    nonlin_f = phi * cp.log(phi) + (epsilon / 2.0) * phi**2 + (g / 3.0) * phi**3 + (1 / 4.0) * phi**4

    f = phi * lin_f + nonlin_f

    return f.get().mean()  # Return as a NumPy scalar

def BuildCupyGPU_Simulation(modelParameters, geometry):
  simulation = Simulation(
    calc_nonlin_mu_impl,
    calc_mu_terms_kspace_impl,
    calc_phi_next_impl)
  simulation.Model = modelParameters
  simulation.Geometry = geometry

  # construct derivative kernels
  derivativeKernels = BuildDerivativeKernels(geometry)
  simulation.State = _SimulationState(modelParameters, derivativeKernels)

  # views of primary fields for user convenience
  simulation.phi = simulation.State.phi
  simulation.mu = simulation.State.mu

  return simulation

def calc_nonlin_mu_impl(self: Simulation):
  """
  Calculates the nonlinear term of the chemical potential μ in real space.
  """
  phi = self.State.phi
  nonlin_mu = self.State.nonlin_mu
  g = self.Model.g
  minPhi = self.Model.minPhi
  minLog = self.Model.minLog

  # compute nonlinear chemical potential μ = logϕ + gϕ² + ϕ³
  # with np.errstate(divide='ignore', invalid='ignore'):
  #   nonlin_mu[...] = cp.where(phi > minPhi, cp.log(phi), minLog) + g * phi**2 + phi**3
  nonlin_mu[...] = cp.log(phi) + g * phi**2 + phi**3

def calc_mu_terms_kspace_impl(self: Simulation):
  """
  Calculates μ, ⟨μ⟩ in k-space.
  """
  # Calc μ (k-space):
  kernel_lin_mu = self.State.kernel_lin_mu
  phi_hat = self.State.phi_hat
  nonlin_mu_hat = self.State.nonlin_mu_hat
  mu_hat = self.State.mu_hat
  mu_hat[...] = kernel_lin_mu * phi_hat + nonlin_mu_hat

  # Calc ⟨μ⟩ (k-space):
  kernel_gaussian = self.State.kernel_gaussian
  local_mu_hat = self.State.local_mu_hat
  local_mu_hat[...] = kernel_gaussian * mu_hat

def calc_phi_next_impl(self: Simulation):
  """
  Calculates the next time step of the density field φ in k-space.
  """
  Gamma = self.Model.Gamma
  kernel_linexp_phidt = self.State.kernel_linexp_phidt
  kernel_nonlinexp_phidt = self.State.kernel_nonlinexp_phidt
  kernel_laplacian = self.State.kernel_laplacian
  phi_hat = self.State.phi_hat
  nonlin_mu_hat = self.State.nonlin_mu_hat
  phi_next_hat = self.State.phi_next_hat
  phi_next_hat[...] = (
    kernel_linexp_phidt * phi_hat +
    kernel_nonlinexp_phidt * (
      Gamma * kernel_laplacian * nonlin_mu_hat
    )
  )
