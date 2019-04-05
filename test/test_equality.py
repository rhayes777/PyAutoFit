from copy import deepcopy

from autofit import mock
from autofit.mapper import prior as p
from autofit.mapper import prior_model as pm
from autofit.mapper import model_mapper as mm
import pytest


@pytest.fixture(name="prior_model")
def make_prior_model():
    return pm.PriorModel(mock.GeometryProfile)


class TestCase(object):
    def test_prior_model(self, prior_model):
        prior_model_copy = deepcopy(prior_model)
        assert prior_model == prior_model_copy

        prior_model_copy.centre_0 = p.UniformPrior()

        assert prior_model != prior_model_copy

    def test_list_prior_model(self, prior_model):
        list_prior_model = pm.ListPriorModel([prior_model])
        list_prior_model_copy = deepcopy(list_prior_model)
        assert list_prior_model == list_prior_model_copy

        list_prior_model[0].centre_0 = p.UniformPrior()

        assert list_prior_model != list_prior_model_copy

    def test_model_mapper(self, prior_model):
        model_mapper = mm.ModelMapper()
        model_mapper.prior_model = prior_model
        model_mapper_copy = deepcopy(model_mapper)

        assert model_mapper == model_mapper_copy

        model_mapper.prior_model.centre_0 = p.UniformPrior()

        assert model_mapper != model_mapper_copy
