import pickle

from autofit import mock
from autofit.optimize import non_linear as nl


class TestCase(object):
    def test_simple_pickle(self):
        optimiser = nl.MultiNest("phasename")
        optimiser.variable.profile = mock.EllipticalProfile
        pickled_optimiser = pickle.loads(pickle.dumps(optimiser))

        assert optimiser.variable.priors == pickled_optimiser.variable.priors
        assert optimiser.phase_path == pickled_optimiser.phase_path
