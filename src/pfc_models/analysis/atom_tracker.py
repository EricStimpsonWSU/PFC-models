"""Locate and track atoms by Gaussian fitting of density peaks.

Provenance: log/gpu-spectral-extV-implementation/_atomTracker.py - the live
copy, imported by four extV notebooks. Its `_AtomTracker` is structurally
identical to the one inside `_trackers.py`, which nothing imports and which is
not migrated.
"""

# numerics
from dataclasses import dataclass

import numpy as np

# fit atom_neighborhood to isometric 2D gaussian + background
from scipy.optimize import curve_fit

# single/double precision types
from pfc_models.backends.numpy_fftw import npFloatType


@dataclass
class FitParameters:
  coords: tuple[float, float]
  A0: float
  sigma: float
  b: float

  def __str__(self):
    return (
      f"  coords: {self.coords}\n"
      f"  A0:     {self.A0:.4f}\n"
      f"  sigma:  {self.sigma:.4f}\n"
      f"  b:      {self.b:.4f}"
    )


@dataclass
class FitResult:
  inputCoords: tuple[float, float]
  outputParameters: FitParameters
  chi2: float
  pcov: np.ndarray
  atomNeighborhood: np.ndarray
  X_Neighborhood: np.ndarray
  Y_Neighborhood: np.ndarray

  def __str__(self):
    return (
      f"FitResult:\n"
      f"• inputCoords: {self.inputCoords}\n"
      f"• outputParameters:\n{self.outputParameters}\n"
      f"• chi2: {self.chi2:.6f}\n"
      f"• pcov: [{self.pcov.shape}]\n"
      f"• atomNeighborhood: [{self.atomNeighborhood.shape}]\n"
      f"• X_Neighborhood: [{self.X_Neighborhood.shape}]\n"
      f"• Y_Neighborhood: [{self.Y_Neighborhood.shape}]"
    )


class _AtomTracker:
  def __init__(self, geometry):
    self.Geometry = geometry

    # current positions of tracked atoms: [[x1, y1], [x2, y2], ...]
    self.atomCoords = []

    # list of lists containing (x, y, vx, vy) tuples for each atom
    self.atomTrajectories = []

  def LocateAtom(self, phi, x0, y0):
    """
    Locate atom in the local density field using estimated starting position.
    """
    # Find the nearest grid point to the estimated position
    nx, ny = self.Geometry.x_to_ix(x0), self.Geometry.y_to_iy(y0)

    # Extract a 5x5 neighborhood around the estimated position
    neighborhood = phi[ny - 2 : ny + 3, nx - 2 : nx + 3]

    # Find the maximum in the neighborhood
    max_idx = np.argmax(neighborhood)
    dy_max, dx_max = np.unravel_index(max_idx, neighborhood.shape)
    nx += dx_max - 2
    ny += dy_max - 2

    # Perform parabolic interpolation to refine the position
    neighborhood_3x3 = phi[ny - 1 : ny + 2, nx - 1 : nx + 2]

    phi_mid = neighborhood_3x3[1, 1]
    phi_left = neighborhood_3x3[1, 0]
    phi_right = neighborhood_3x3[1, 2]
    phi_up = neighborhood_3x3[2, 1]
    phi_down = neighborhood_3x3[0, 1]

    # x0, y0 to parabolic interpolation max
    denom_x = 2 * (phi_left + phi_right - 2 * phi_mid)
    delta_x = -(phi_right - phi_left) / denom_x if denom_x != 0 else 0.0

    denom_y = 2 * (phi_up + phi_down - 2 * phi_mid)
    delta_y = -(phi_up - phi_down) / denom_y if denom_y != 0 else 0.0

    nx += delta_x
    ny += delta_y
    x0 = self.Geometry.px_to_x(nx)
    y0 = self.Geometry.py_to_y(ny)
    return x0, y0

  def FitAtomToGaussian(self, phi, x0, y0):
    """
    Fit an atom to a Gaussian plus background.
    """

    def fit_func(xy_grid, x0, y0, A0, sigma0, b0):
      x, y = xy_grid

      return b0 + A0 * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma0**2))

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
      fit_func,
      xy_grid_flat,
      atomNeighborhood.ravel(),
      p0=[x0, y0, A0, sigma0, b0],
    )

    chi2 = np.sum((fit_func(xy_grid_flat, *popt) - atomNeighborhood.ravel()) ** 2)
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

  def AddAtom(self, phi, x0, y0):
    x, y = self.LocateAtom(phi, x0, y0)
    self.atomCoords.append([0, x, y])

    # Initialize trajectory with 0.0 initial velocity: (x, y, vx, vy)
    self.atomTrajectories.append([(0, x, y, 0.0, 0.0)])

  def TrackAtoms(self, phi, deltaT):
    """
    Track the position and velocity of atoms in the local density field.
    """

    if len(self.atomCoords) == 0:
      print("No atoms to track.")
      return

    for i, (t, x_old, y_old) in enumerate(self.atomCoords):
      # 1. Locate the new position
      x_new, y_new = self.LocateAtom(phi, x_old, y_old)

      # 2. Calculate the component velocities
      vx = (x_new - x_old) / deltaT
      vy = (y_new - y_old) / deltaT

      # 3. Update the persistent current coordinates
      self.atomCoords[i] = [t + deltaT, x_new, y_new]

      # 4. Append the requested 4-tuple to this atom's trajectory list
      self.atomTrajectories[i].append((t + deltaT, x_new, y_new, vx, vy))
