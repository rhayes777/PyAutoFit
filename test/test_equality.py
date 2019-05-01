from copy import deepcopy

import pytest

import autofit.mapper.model
from autofit import mock
from autofit.mapper import model_mapper as mm
from autofit.mapper import prior as p
from autofit.mapper import prior_model as pm
from autofit.optimize import non_linear


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
        list_prior_model = pm.CollectionPriorModel([prior_model])
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

    def test_non_trivial_equality(self):
        model_mapper = mm.ModelMapper()
        model_mapper.galaxy = mock.GalaxyModel(light_profile=mock.GeometryProfile, mass_profile=mock.GeometryProfile)
        model_mapper_copy = deepcopy(model_mapper)

        assert model_mapper == model_mapper_copy

        model_mapper.galaxy.light_profile.centre_0 = p.UniformPrior()

        assert model_mapper != model_mapper_copy

    def test_model_instance_equality(self):
        model_instance = autofit.mapper.model.ModelInstance()
        model_instance.profile = mock.GeometryProfile()
        model_instance_copy = deepcopy(model_instance)

        assert model_instance == model_instance_copy

        model_instance.profile.centre = (1., 2.)

        assert model_instance != model_instance_copy

    def test_non_linear_equality(self):
        nlo = non_linear.NonLinearOptimizer("phase name")
        nlo.variable.profile = mock.GeometryProfile
        nlo_copy = deepcopy(nlo)

        assert nlo_copy == nlo

        nlo.variable.profile.centre_0 = p.UniformPrior()

        assert nlo_copy != nlo

    def test_multinest_equality(self):
        nlo = non_linear.MultiNest("phase name")
        nlo_copy = deepcopy(nlo)

        assert nlo == nlo_copy

        nlo.n_live_points += 1

        assert nlo != nlo_copy
