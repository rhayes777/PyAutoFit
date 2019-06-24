import pickle


import test.mock
from test import mock
import autofit as af


class TestCase(object):
    def test_simple_pickle(self):
        optimiser = af.MultiNest("phasename")
        optimiser.variable.profile = test.mock.EllipticalProfile
        pickled_optimiser = pickle.loads(pickle.dumps(optimiser))

        assert optimiser.variable.priors == pickled_optimiser.variable.priors
        assert optimiser.phase_path == pickled_optimiser.phase_path
