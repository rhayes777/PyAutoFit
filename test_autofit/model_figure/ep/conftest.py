import itertools

import pytest

import autofit as af
from autofit.tools.namer import namer


@pytest.fixture(autouse=True)
def reset_ids():
    """
    Model object and prior ids are global counters, and so is ``namer`` -- the
    source of a declarative factor's name (``AnalysisFactor0``,
    ``HierarchicalFactor0``), which the EP spec records.  They are reset before
    every test so that the spec is a function of the factor graph alone, which
    is what the determinism assertions rely on.
    """
    af.ModelObject._ids = itertools.count()
    af.Prior._ids = itertools.count()
    namer.reset()
