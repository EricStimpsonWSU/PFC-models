"""External potential shapes for logpfc_extv: generalised Gaussian, Gaussian
dipole, tanh double-vacancy.

Provenance: log/gpu-spectral-extV-implementation/_externalPotentials.py. Model
code rather than substrate: these are the potential forms this model
integrates, and no other model imports them. Its `FitParameters`/`FitResult`
differ from the ones in `analysis.atom_tracker`; both are kept where they are.
"""
# numerics
from dataclasses import dataclass

# numerics
import cupy as cp
import numpy as np

# fit atom_neighborhood to isometric 2D gaussian + background
from scipy.optimize import curve_fit

# single/double precision types
from pfc_models.backends.cupy_cuda import cpFloatType


@dataclass
class FitParameters:
  coords: tuple
  A0: float
  sigma: float
  b: float

@dataclass
class FitResult:
  inputCoords: tuple
  outputParameters: FitParameters
  chi2: float
  pcov: np.ndarray
  atomNeighborhood: np.ndarray
  X_Neighborhood: np.ndarray
  Y_Neighborhood: np.ndarray


def GeneralizedGaussian(xy_grid, x0, y0, A0, sigma0, b0):
  x, y = xy_grid

  return b0 + A0 * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma0**2))


class _ExternalPotential:
  """
  Builds sub-pixel precse external potential V(x,y) and its Fourier transform V_hat(kx,ky) on the GPU
  for a variety of potential functions V_func(x,y) and parameters V0, width, x1, x2.
  """
  def __init__(self, geometry):
    self.Geometry = geometry
    self.V = cp.zeros(geometry.X.shape, dtype=cpFloatType)
    self.V_hat = cp.fft.rfft2(self.V)
    self.lap_V = cp.zeros(geometry.X.shape, dtype=cpFloatType)  # ∇²V in real space
    self.lap_V_hat = cp.fft.rfft2(self.lap_V)  # ∇²V in k-space

    self.X = cp.asarray(self.Geometry.X)
    self.Y = cp.asarray(self.Geometry.Y)

  def FitAtomToGaussian(self, phi, x0, y0):
    """
    Fit an atom to a Gaussian plus background.
    """

    a = self.Geometry.a
    left_ix = self.Geometry.x_to_ix(x0 - a / 2)
    right_ix = self.Geometry.x_to_ix(x0 + a / 2)
    bottom_ix = self.Geometry.y_to_iy(y0 - a / 2)
    top_ix = self.Geometry.y_to_iy(y0 + a / 2)
    print(
      f"Fitting atom at ({x0:.3f}, {y0:.3f}) with neighborhood indices: left={left_ix}, right={right_ix}, bottom={bottom_ix}, top={top_ix}"
    )

    atomNeighborhood = phi[bottom_ix : top_ix + 1, left_ix : right_ix + 1]
    X_Neighborhood = self.Geometry.X[bottom_ix : top_ix + 1, left_ix : right_ix + 1]
    Y_Neighborhood = self.Geometry.Y[bottom_ix : top_ix + 1, left_ix : right_ix + 1]
    print(f"Atom neighborhood shape: {atomNeighborhood.shape}")

    xy_grid_flat = (X_Neighborhood.ravel(), Y_Neighborhood.ravel())

    b0 = atomNeighborhood.min()
    A0 = atomNeighborhood.max() - atomNeighborhood.min()
    sigma0 = a / 4

    popt, pcov = curve_fit(
      GeneralizedGaussian,
      xy_grid_flat,
      atomNeighborhood.ravel(),
      p0=[x0, y0, A0, sigma0, b0],
    )

    chi2 = np.sum((GeneralizedGaussian(xy_grid_flat, *popt) - atomNeighborhood.ravel()) ** 2)
    xR, yR, A0, sigma0, b0 = popt

    result = FitResult(
      inputCoords=(x0, y0),
      outputParameters=FitParameters(coords=(xR, yR), A0=A0, sigma=sigma0, b=b0),
      chi2=chi2,
      pcov=pcov,
      atomNeighborhood=atomNeighborhood,
      X_Neighborhood=X_Neighborhood,
      Y_Neighborhood=Y_Neighborhood,
    )

    return result

  def SetGaussianDipolePotential(self, V0, X0, X1, sigma):
    """
    Set a Gaussian dipole potential with positive peak at X0 and negative peak at X1, with width sigma.
    """
    x0, y0 = X0
    x1, y1 = X1
    self.V[...] = V0 * (
      cp.exp(-((self.X - x0) ** 2 + (self.Y - y0) ** 2) / (2 * sigma**2))
      - cp.exp(-((self.X - x1) ** 2 + (self.Y - y1) ** 2) / (2 * sigma**2))
    )
    self.V_hat[...] = cp.fft.rfft2(self.V)

  def SetPotential(self, V0, V_func):
    self.V[...] = V0 * cp.asarray(V_func)
    self.V_hat[...] = cp.fft.rfft2(self.V)

  def SetPotential_alongX(self, V0, width, x1, x2):
    self.V[...] = V0 * cp.tanh((cp.abs(self.X - (x1 + x2) / 2) - (x2 - x1)/2)/width)
    self.V_hat[...] = cp.fft.rfft2(self.V)

  def SetDoubleVacancyAlongX_TanhPotential(self, x1, x2, y1, y2, width_x, width_y, V0):
    self.V[...] = V0 * (cp.tanh((cp.abs(self.X-(x1 + x2) / 2) - (x2 - x1)/2)/width_x) - 1) / 2 * (1 - cp.tanh((cp.abs(self.Y-(y1 + y2) / 2) - (y2 - y1)/2)/width_y)) / 2 + 1 # x-ripple
    self.V_hat[...] = cp.fft.rfft2(self.V)
