import itertools

import pytest

import autofit as af


@pytest.fixture(autouse=True)
def reset_ids():
    """
    Model object and prior ids are global counters, and the spec records both.
    Reset them before every test so ids -- and therefore the serialised spec --
    are a function of the model alone (copied from ``test_autofit/test_visualise.py``).
    """
    af.ModelObject._ids = itertools.count()
    af.Prior._ids = itertools.count()


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
