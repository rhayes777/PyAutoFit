from autofit import mock
from autofit.optimize import non_linear as nl
import pickle


class TestCase(object):
    def test_simple_pickle(self):
        optimiser = nl.MultiNest("phasename")
        optimiser.variable.profile = mock.EllipticalProfile

        assert optimiser.variable.priors == pickle.loads(pickle.dumps(optimiser)).variable.priors
