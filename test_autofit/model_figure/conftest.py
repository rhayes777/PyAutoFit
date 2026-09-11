import itertools

import pytest

import autofit as af


@pytest.fixture(autouse=True)
def reset_ids():
    """
    Model object and prior ids are global counters and the spec records both, so
    they are reset before every test -- the figure is then a function of the
    model alone, which is what the determinism acceptance asserts.
    """
    af.ModelObject._ids = itertools.count()
    af.Prior._ids = itertools.count()
