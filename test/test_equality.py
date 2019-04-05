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

    def test_list_prior_model(self):
        list_prior_model = pm.ListPriorModel([pm.PriorModel(mock.GeometryProfile)])
        list_prior_model_copy = deepcopy(list_prior_model)
        assert list_prior_model == list_prior_model_copy

        list_prior_model[0].centre_0 = p.UniformPrior()

        assert list_prior_model != list_prior_model_copy
