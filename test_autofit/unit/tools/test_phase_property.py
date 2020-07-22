import os

import pytest

from autoconf import conf
import autofit as af
from autofit.non_linear import abstract_search
from test_autofit import mock
from test_autofit import mock_real

directory = os.path.dirname(os.path.realpath(__file__))


class NLO(abstract_search.NonLinearSearch):

    @property
    def config_type(self):
        return conf.instance.mock

    @property
    def tag(self):
        return"nlo"

    def _fit(self, model, analysis):
        class Fitness:
            def __init__(self, instance_from_vector, instance):
                self.result = None
                self.instance_from_vector = instance_from_vector
                self.instance = instance

            def __call__(self, vector):
                instance = self.instance_from_vector(vector)
                for key, value in self.instance.__dict__.items():
                    setattr(instance, key, value)

                log_likelihood = analysis.log_likelihood_function(instance)
                self.result = abstract_search.Result(
                    instance, log_likelihood
                )

                # Return Chi squared
                return -2 * log_likelihood

        fitness_function = Fitness(
            model.instance_from_vector, model.instance_from_prior_medians
        )
        fitness_function(model.prior_count * [0.5])

        return fitness_function.result


@pytest.fixture(name="phase")
def make_phase():
    return MyPhase(af.Paths(name=""), search=NLO)


class MyPhase(af.AbstractPhase):
    prop = af.PhaseProperty("prop")


@pytest.fixture(name="list_phase")
def make_list_phase():
    return MyPhase(af.Paths(name=""), search=NLO)


class TestPhasePropertyList:
    def test_classes(self, list_phase):
        objects = [mock_real.GalaxyModel(), mock_real.GalaxyModel()]

        list_phase.prop = objects

        assert list_phase.model.prop == objects
        assert list_phase.prop == objects

    def test_abstract_prior_models(self, list_phase):
        objects = [af.AbstractPriorModel(), af.AbstractPriorModel()]

        list_phase.prop = objects

        assert list_phase.model.prop == objects
        assert list_phase.prop == objects

    def test_mix(self, list_phase):
        objects = [mock_real.GalaxyModel(), mock.MockComponents()]

        list_phase.prop = objects

        assert list_phase.model.prop == objects
        assert list_phase.prop == objects

    def test_set_item(self, list_phase):
        galaxy_prior_0 = mock_real.GalaxyModel()
        objects = [galaxy_prior_0, mock.MockComponents()]

        list_phase.prop = objects

        galaxy_prior_1 = mock_real.GalaxyModel()
        list_phase.prop[1] = galaxy_prior_1

        assert list_phase.model.prop == [galaxy_prior_0, galaxy_prior_1]

        galaxy = mock.MockComponents()

        list_phase.prop[0] = galaxy
        assert list_phase.prop == [galaxy, galaxy_prior_1]


class TestPhasePropertyCollectionAttributes:
    def test_set_list_as_dict(self, list_phase):
        galaxy_model = mock_real.GalaxyModel()
        list_phase.prop = dict(one=galaxy_model)

        assert len(list_phase.prop) == 1
        # noinspection PyUnresolvedReferences
        assert list_phase.prop.one == galaxy_model

    def test_override_property(self, list_phase):
        galaxy_model = mock_real.GalaxyModel()

        list_phase.prop = dict(one=mock_real.GalaxyModel())

        list_phase.prop.one = galaxy_model

        assert len(list_phase.prop) == 1
        assert list_phase.prop.one == galaxy_model

    def test_named_list_items(self, list_phase):
        galaxy_model = mock_real.GalaxyModel()
        list_phase.prop = [galaxy_model]

        # noinspection PyUnresolvedReferences
        assert getattr(list_phase.prop, "0") == galaxy_model

    def test_mix(self, list_phase):
        objects = dict(one=mock_real.GalaxyModel(), two=mock.MockComponents())

        list_phase.prop = objects

        list_phase.prop.one = mock.MockComponents()

        assert len(list_phase.model.prop) == 2

    def test_named_attributes_in_model(self, list_phase):
        galaxy_model = mock_real.GalaxyModel(model_redshift=True)
        list_phase.prop = dict(one=galaxy_model)

        assert list_phase.model.prior_count == 1
        assert list_phase.model.prop.one == galaxy_model

        instance = list_phase.model.instance_from_prior_medians()

        assert instance.prop.one is not None
        assert len(instance.prop) == 1

    def test_named_attributes_in_model_override(self, list_phase):
        list_phase.prop = dict(one=mock_real.GalaxyModel())

        assert list_phase.model.prior_count == 0

        galaxy_model = mock_real.GalaxyModel(model_redshift=True)

        list_phase.prop.one = galaxy_model

        assert list_phase.model.prior_count == 1
        # assert list_phase.model.one == galaxy_model

        instance = list_phase.model.instance_from_prior_medians()

        # assert instance.one is not None
        assert len(instance.prop) == 1

    def test_named_attributes_in_instance(self, list_phase):
        galaxy = mock.MockComponents()
        list_phase.prop = dict(one=galaxy)

        assert list_phase.model.prior_count == 0
        assert list_phase.model.prop.one == galaxy

    def test_shared_priors(self, list_phase):
        list_phase.prop = dict(
            one=mock_real.GalaxyModel(model_redshift=True), two=mock_real.GalaxyModel(model_redshift=True)
        )

        assert list_phase.model.prior_count == 2

        # noinspection PyUnresolvedReferences
        list_phase.prop.one.redshift = list_phase.prop.two.redshift

        assert list_phase.model.prior_count == 1

    def test_hasattr(self, list_phase):
        list_phase.prop = dict()

        assert not hasattr(list_phase.prop, "one")
        list_phase.prop = dict(one=mock_real.GalaxyModel(model_redshift=True))

        assert hasattr(list_phase.prop, "one")

    def test_position_not_a_prior(self, list_phase):
        list_phase.prop = [af.PriorModel(mock.MockComponents)]

        assert list_phase.model.prior_count == 1
        assert "parameter" == list_phase.model.prior_tuples_ordered_by_id[0][0]

        prior_model = af.PriorModel(mock.MockComponents)
        prior_model.phase_property_position = 0

        print(prior_model.instance_tuples)
        assert len(prior_model.instance_tuples) == 0
