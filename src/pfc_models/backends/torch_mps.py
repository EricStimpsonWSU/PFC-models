"""PyTorch MPS (Apple GPU) backend.

Sibling of the other modules in `pfc_models.backends`. Every backend module
exposes the same names so a model can select one at construction time:

    xp          array module (numpy, cupy, torch, ...)
    fftn/ifftn  forward/inverse n-D FFT on real input
    rfftn/irfftn
    to_numpy(a) host copy as a numpy array
    available() -> bool
    name        str

Nothing here is a base class and nothing imports another backend. Add a new
backend by adding a sibling file that satisfies the same names.

STUB — not implemented. Planned: Mac GPU target. Note torch's rfftn semantics differ slightly from numpy's; document when implemented.
"""

name = "torch_mps"


def available() -> bool:
    raise NotImplementedError("torch_mps backend not implemented yet")
