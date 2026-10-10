"""Regression test for logpfc: fixed seed, asserted free energy at a fixed step.

The point is not accuracy — it is detecting the day a refactor silently changes
the physics. Write this before migrating the model, and record the reference
value from the ORIGINAL implementation, not from the migrated one.
"""
import pytest

pytestmark = pytest.mark.skip(reason="logpfc not migrated yet")

REFERENCE_FREE_ENERGY = None   # fill from the original implementation
REFERENCE_STEP = 1000
RTOL = 1e-10


def test_free_energy_at_fixed_step():
    raise NotImplementedError
