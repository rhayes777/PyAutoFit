import pickle

import test_autofit
import autofit as af


class TestCase(object):
    def test_simple_pickle(self):
        optimiser = af.MultiNest("phasename")
        optimiser.variable.profile = test_autofit.mock.EllipticalProfile
        pickled_optimiser = pickle.loads(pickle.dumps(optimiser))

        assert optimiser.variable.priors == pickled_optimiser.variable.priors
        assert optimiser.phase_path == pickled_optimiser.phase_path
