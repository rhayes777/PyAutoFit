from copy import deepcopy

from autofit import mock
from autofit.mapper import prior as p
from autofit.mapper import prior_model as pm


class TestCase(object):
    def test_prior_model(self):
        prior_model = pm.PriorModel(mock.GeometryProfile)
        prior_model_copy = deepcopy(prior_model)
        assert prior_model == prior_model_copy

        prior_model_copy.centre_0 = p.UniformPrior()

        assert prior_model != prior_model_copy
