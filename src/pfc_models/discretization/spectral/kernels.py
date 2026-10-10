"""Spectral operator kernels: derivatives, Gaussian smoothing, ETD time-stepping.

STUB. Migration notes:

- The existing forks disagree on the Gaussian kernel's *name*
  (`Get_kGaussian` in twelve copies, `GetGaussianKernal_k` in two). Pick one
  here; `kernel_gaussian` is suggested.
- `GetLinearKernel_mu` takes an extra `M=0` term in the external-potential
  fork. That is a model concern, not a substrate concern — keep the substrate
  kernel pure and let the model add its term.
- `GetExponentialTimesteppingKernels` has five versions across the forks; it is
  the single largest unexplained divergence in the audit. Reconcile before
  migrating.
"""

raise NotImplementedError("spectral.kernels not migrated yet")
