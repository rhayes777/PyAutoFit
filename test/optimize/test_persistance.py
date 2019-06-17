import pickle

import autofit.optimize.non_linear.multi_nest
import test.mock
from test import mock
from autofit.optimize import non_linear as nl


class TestCase(object):
    def test_simple_pickle(self):
        optimiser = autofit.optimize.non_linear.multi_nest.MultiNest("phasename")
        optimiser.variable.profile = test.mock.EllipticalProfile
        pickled_optimiser = pickle.loads(pickle.dumps(optimiser))

        assert optimiser.variable.priors == pickled_optimiser.variable.priors
        assert optimiser.phase_path == pickled_optimiser.phase_path
