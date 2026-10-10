"""2D periodic spectral geometry — the reference implementation of the Geometry
contract. Siblings (`geometry_1d`, `geometry_3d`, `finitediff.geometry_2d`)
repeat these names.

Contract
--------
    shape       tuple[int, ...]      grid shape
    dx          tuple[float, ...]    spacing per axis
    L           tuple[float, ...]    box length per axis
    X, Y, ...   real-space coordinate arrays
    KX, KY, ... wavenumber arrays matching the rfft layout
    K2          |k|^2
    x_to_ix(x), ix_to_x(i), ...      coordinate/index conversions
    periodic    tuple[bool, ...]     (True, True) here

A model written against these names runs on any geometry satisfying them.

STUB — migrate from Build2DGeometry / Geometry in the existing forks. Their
three versions differ; reconcile before implementing and record the choice in
docs/ARCHITECTURE.md.
"""

raise NotImplementedError("spectral.geometry_2d not migrated yet")
