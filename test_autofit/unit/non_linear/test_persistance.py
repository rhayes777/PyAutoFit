import pickle

import autofit as af
from autofit.non_linear.paths import Paths


class TestCase:
    def test_simple_pickle(self):
        optimiser = af.DynestyStatic(paths=Paths("phasename"))
        pickled_optimiser = pickle.loads(pickle.dumps(optimiser))
        assert optimiser.paths.path_prefix == pickled_optimiser.paths.path_prefix
