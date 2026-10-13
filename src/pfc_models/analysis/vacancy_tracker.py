"""Locate and track vacancies by sub-pixel minimisation of the local density.

Provenance: `_VacancyTracker` inlined in
log/gpu-spectral-extV-implementation/Simulation.py - the live copy, the one
`BuildCupyGPU_Simulation` constructs. Two other copies exist in the source
tree: `Simulation2ndOrder.py` carries a structurally identical class (that
fork stays in the sandbox), and `_trackers.py` carries a third design with a
divergent `LocateVacancy` and an `AddAtom`/`TrackAtoms` API that nothing
imports. Needs no fit dataclasses: it has no `FitAtomToGaussian`.
"""

# numerics
import numpy as np


class _VacancyTracker:
  def __init__(self, geometry):
    self.Geometry = geometry
    self.vacancyCoords = []
    self.vacancyTrajectories = []

  def LocateVacancy(self, local_phi, x0, y0):
    """
    Locate vacancies in the local density field using a threshold and minimum distance.
    """

    # from (x0,y0) in real-space coords to nearest (nx0, ny0) in pixel coordinates using
    # geometrie's X, Y mesh
    nx = int(np.round(x0 / self.Geometry.dx))
    ny = int(np.round(y0 / self.Geometry.dy))

    # find absolute minimum in a 3x3 neighborhood
    neighborhood_3x3 = local_phi[ny - 1 : ny + 2, nx - 1 : nx + 2]
    min_idx = np.argmin(neighborhood_3x3)
    dy, dx = np.unravel_index(min_idx, neighborhood_3x3.shape)

    # adjust pixel coords of vacancy
    ny += dy - 1
    nx += dx - 1

    # extract cross-kernel values
    f_mid = local_phi[ny, nx]
    f_left = local_phi[ny, nx - 1]
    f_right = local_phi[ny, nx + 1]
    f_up = local_phi[ny - 1, nx]
    f_down = local_phi[ny + 1, nx]

    # analytic 2D sub-pixel calc
    denom_x = 2.0 * (f_left - 2.0 * f_mid + f_right)
    delta_x = (f_left - f_right) / denom_x if denom_x != 0 else 0.0

    denom_y = 2.0 * (f_up - 2.0 * f_mid + f_down)
    delta_y = (f_up - f_down) / denom_y if denom_y != 0 else 0.0

    nx += delta_x
    ny += delta_y

    # get (x,y) in real-space coordinates
    x = nx * self.Geometry.dx
    y = ny * self.Geometry.dy
    return (x, y)

  def AddVacancy(self, local_phi, nx, ny):
    x, y = self.LocateVacancy(local_phi, nx * self.Geometry.dx, ny * self.Geometry.dy)
    self.vacancyCoords.append((x, y))
    self.vacancyTrajectories.append(
      [(x, y)]
    )  # Initialize trajectory with the first position

  def TrackVacancies(self, local_phi):
    """
    Track the positions of vacancies in the local density field.
    """
    for i, (x0, y0) in enumerate(self.vacancyCoords):
      x, y = self.LocateVacancy(local_phi, x0, y0)
      self.vacancyCoords[i] = (x, y)
      self.vacancyTrajectories[i].append((x, y))  # Append new position to trajectory
