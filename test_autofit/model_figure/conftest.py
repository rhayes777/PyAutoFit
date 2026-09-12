import itertools

import pytest

import autofit as af
from autofit.tools.namer import namer


@pytest.fixture(autouse=True)
def reset_ids():
    """
    Model object and prior ids are global counters and the spec records both, so
    they are reset before every test -- the figure is then a function of the
    model alone, which is what the determinism acceptance asserts.

    ``namer`` is reset for the same reason: a declarative factor's name
    (``AnalysisFactor0``, ``HierarchicalFactor0``) comes from that global
    counter, and the graphical pass records it.
    """
    af.ModelObject._ids = itertools.count()
    af.Prior._ids = itertools.count()
    namer.reset()
