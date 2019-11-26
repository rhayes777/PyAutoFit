import pickle

import autofit as af
from autofit import Paths


class TestCase(object):
    def test_simple_pickle(self):
        optimiser = af.MultiNest(Paths("phasename"))
        pickled_optimiser = pickle.loads(pickle.dumps(optimiser))
        assert optimiser.paths.phase_path == pickled_optimiser.paths.phase_path
