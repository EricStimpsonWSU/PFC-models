"""2D periodic spectral geometry - the reference implementation of the Geometry
contract. Siblings (`geometry_1d`, `geometry_3d`, `finitediff.geometry_2d`)
repeat these names.

Contract
--------
    Nx, Ny          grid shape
    dx, dy          spacing per axis
    Lx, Ly          box length per axis
    x, y, X, Y      real-space coordinates and mesh
    kx, ky, KX, KY  wavenumbers; kx uses rfftfreq, ky uses fftfreq
    K2              |k|^2
    px_to_x, py_to_y, x_to_px, y_to_py, x_to_ix, y_to_iy
                    coordinate/index conversions
    periodic        (True, True) - implicit in the spectral operators

A model written against these names runs on any geometry satisfying them.

Provenance
----------
`Geometry` from log/gpu-spectral-extV-implementation/Simulation.py: the base
forks carry a degenerate `__init__` returning None, while the extV copy declares
the attributes and implements the six coordinate/index conversions the contract
names. `Build2DGeometry` from log/gpu-spectral-base-implementation: identical to
the extV copy apart from the two triangular-lattice constants `a` and `h`, which
only extV reads and which are attached in `models/logpfc_extv/` instead. `K2` is
added here because the contract names it; no fork defined it, and the models keep
their inline `KX**2 + KY**2` expressions.
"""

# numerics
import numpy as np

# host dtype aliases
from pfc_models.backends.numpy_fftw import npFloatType


class Geometry:
  def __init__(self):
    self.Nx = None
    self.Ny = None
    self.Lx = None
    self.Ly = None
    self.dx = None
    self.dy = None
    self.x = None
    self.y = None
    self.X = None
    self.Y = None

  def px_to_x(self, px):
    return px * self.dx

  def py_to_y(self, py):
    return py * self.dy

  def x_to_px(self, x):
    return x / self.dx

  def y_to_py(self, y):
    return y / self.dy

  def x_to_ix(self, x):
    return int(x / self.dx)

  def y_to_iy(self, y):
    return int(y / self.dy)


def Build2DGeometry(Nx, Ny, Lx, Ly):
  geometry = Geometry()
  geometry.Nx = Nx
  geometry.Ny = Ny
  geometry.Lx = Lx
  geometry.Ly = Ly
  geometry.dx = geometry.Lx / geometry.Nx
  geometry.dy = geometry.Ly / geometry.Ny

  geometry.x = np.linspace(0, geometry.Lx, geometry.Nx, endpoint=False, dtype=npFloatType)
  geometry.y = np.linspace(0, geometry.Ly, geometry.Ny, endpoint=False, dtype=npFloatType)
  geometry.X, geometry.Y = np.meshgrid(geometry.x, geometry.y)

  geometry.kx = 2 * np.pi * np.fft.rfftfreq(geometry.Nx, d=geometry.dx).astype(npFloatType)
  geometry.ky = 2 * np.pi * np.fft.fftfreq(geometry.Ny, d=geometry.dy).astype(npFloatType)
  geometry.KX, geometry.KY = np.meshgrid(geometry.kx, geometry.ky)
  geometry.K2 = geometry.KX**2 + geometry.KY**2

  return geometry
