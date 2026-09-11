import itertools

import pytest

import autofit as af
from autofit.tools.namer import namer


@pytest.fixture(autouse=True)
def reset_ids():
    """
    Model object and prior ids are global counters, and the spec records both.
    Reset them before every test so ids -- and therefore the serialised spec --
    are a function of the model alone (copied from ``test_autofit/test_visualise.py``).

    ``namer`` is reset for the same reason: a declarative factor's name
    (``AnalysisFactor0``, ``HierarchicalFactor0``) comes from that global
    counter, and the graphical pass records it.
    """
    af.ModelObject._ids = itertools.count()
    af.Prior._ids = itertools.count()
    namer.reset()


class NullAnalysis(af.Analysis):
    """The smallest possible analysis: enough to build an ``af.AnalysisFactor``."""

    def log_likelihood_function(self, instance):
        return 0.0


class FwhmLatent(af.Analysis.Latent):
    @staticmethod
    def keys(analysis):
        return ["gaussian.fwhm"]


class LatentAnalysis(NullAnalysis):
    Latent = FwhmLatent
