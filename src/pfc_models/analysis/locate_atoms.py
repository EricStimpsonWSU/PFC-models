"""Single-particle-tracking helpers: locate centres of mass, append tracks.

Provenance: log/gpu-spectral-base-implementation/_locateAtoms.py, identical to
the gpu-spectral-extV copy and absent from the CPU fork. New leaf module: the
scaffold did not anticipate it. Requires h5py and scikit-image.
"""
import h5py
import numpy as np
from scipy.ndimage import center_of_mass
from skimage.feature import peak_local_max


def SPT_LocateCM(phi, geometry, nxy=10, min_distance=5, threshold_abs=0.8):

  # tile the phi field to avoid edge effects in peak finding
  _phiTile = np.tile(phi, (3,3))

  # find peaks in the tiled phi field
  _peaks = peak_local_max(_phiTile, min_distance=int(min_distance), threshold_abs=phi.max()*threshold_abs)

  # loop over peaks and locate CM
  _cmTileList = np.zeros((_peaks.shape[0], 9))
  for i, (y_peak, x_peak) in enumerate(_peaks):
    # define bounding box for CM
    _xB, _xE = max(0, x_peak - nxy//2), min(_phiTile.shape[1], x_peak + nxy//2)
    _yB, _yE = max(0, y_peak - nxy//2), min(_phiTile.shape[0], y_peak + nxy//2)

    # extract region
    _region = _phiTile[_yB:_yE, _xB:_xE]
    _mass = _region.sum()

    # compute center of mass
    _yCM, _xCM = center_of_mass(_region)
    _xCMpx = _xB - geometry.Nx + _xCM
    _yCMpx = _yB - geometry.Ny + _yCM
    _xCM = (_xCMpx) * geometry.dx
    _yCM = (_yCMpx) * geometry.dy
    _cmTileList[i] = [_yCM, _xCM, _yCMpx, _xCMpx, _xB, _xE, _yB, _yE, _mass]
  
  # filter CMs to those in original phi region
  mask = (
    (_cmTileList[:,1] >= 0) & (_cmTileList[:,1] < geometry.Nx * geometry.dx) &
    (_cmTileList[:,0] >= 0) & (_cmTileList[:,0] < geometry.Ny * geometry.dy)
  )

  # print(f'Found {_cmTileList[mask].shape[0]} peaks in phi with min_distance={min_distance}, threshold_abs={threshold_abs}')
  return _cmTileList[mask], mask, _peaks

def SPT_AppendData(cm_list, t, filename):
  # Append current t to cm_list
  cm_list = np.hstack((cm_list, np.full((cm_list.shape[0], 1), t)))

  # Append data to file
  with h5py.File(filename, 'a') as f:
    if 'cm' not in f:
      f.create_dataset('cm', data=cm_list, maxshape=(None, cm_list.shape[1]), dtype='float32') #chunks=True)
    else:
      f['cm'].resize((f['cm'].shape[0] + cm_list.shape[0], cm_list.shape[1]))
      f['cm'][-cm_list.shape[0]:] = cm_list
