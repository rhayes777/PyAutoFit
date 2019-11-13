import os

import pytest

import autofit as af
import autofit.mapper.prior_model.abstract
import autofit.mapper.prior_model.prior_model
import autofit.optimize.non_linear.non_linear
from autofit import Paths
from test_autofit.mock import Galaxy, GalaxyModel

directory = os.path.dirname(os.path.realpath(__file__))

af.conf.instance = af.conf.Config(
    "{}/../../workspace/config".format(directory),
    "{}/../../workspace/output/".format(directory),
)


class NLO(autofit.optimize.non_linear.non_linear.NonLinearOptimizer):
    def fit(self, analysis, model):
        class Fitness(object):
            def __init__(self, instance_from_physical_vector, instance):
                self.result = None
                self.instance_from_physical_vector = instance_from_physical_vector
                self.instance = instance

            def __call__(self, vector):
                instance = self.instance_from_physical_vector(vector)
                for key, value in self.instance.__dict__.items():
                    setattr(instance, key, value)

                likelihood = analysis.fit(instance)
                self.result = autofit.optimize.non_linear.non_linear.Result(
                    instance, likelihood
                )

                # Return Chi squared
                return -2 * likelihood

        fitness_function = Fitness(
            model.instance_from_physical_vector, model.instance_from_prior_medians
        )
        fitness_function(model.prior_count * [0.5])

        return fitness_function.result


@pytest.fixture(name="phase")
def make_phase():
    return MyPhase(Paths(phase_name=""), optimizer_class=NLO)


class MyPhase(af.AbstractPhase):
    prop = af.PhaseProperty("prop")


@pytest.fixture(name="list_phase")
def make_list_phase():
    return MyPhase(Paths(phase_name=""), optimizer_class=NLO)


class TestPhasePropertyList(object):
    def test_classes(self, list_phase):
        objects = [GalaxyModel(), GalaxyModel()]

        list_phase.prop = objects

        assert list_phase.model.prop == objects
        assert list_phase.prop == objects

    def test_abstract_prior_models(self, list_phase):
        objects = [af.AbstractPriorModel(), af.AbstractPriorModel()]

        list_phase.prop = objects

        assert list_phase.model.prop == objects
        assert list_phase.prop == objects

    def test_mix(self, list_phase):
        objects = [GalaxyModel(), Galaxy()]

        list_phase.prop = objects

        assert list_phase.model.prop == objects
        assert list_phase.prop == objects

    def test_set_item(self, list_phase):
        galaxy_prior_0 = GalaxyModel()
        objects = [galaxy_prior_0, Galaxy()]

        list_phase.prop = objects

        galaxy_prior_1 = GalaxyModel()
        list_phase.prop[1] = galaxy_prior_1

        assert list_phase.model.prop == [galaxy_prior_0, galaxy_prior_1]

        galaxy = Galaxy()

        list_phase.prop[0] = galaxy
        assert list_phase.prop == [galaxy, galaxy_prior_1]


class TestPhasePropertyCollectionAttributes(object):
    def test_set_list_as_dict(self, list_phase):
        galaxy_model = GalaxyModel()
        list_phase.prop = dict(one=galaxy_model)

        assert len(list_phase.prop) == 1
        # noinspection PyUnresolvedReferences
        assert list_phase.prop.one == galaxy_model

    def test_override_property(self, list_phase):
        galaxy_model = GalaxyModel()

        list_phase.prop = dict(one=GalaxyModel())

        list_phase.prop.one = galaxy_model

        assert len(list_phase.prop) == 1
        assert list_phase.prop.one == galaxy_model

    def test_named_list_items(self, list_phase):
        galaxy_model = GalaxyModel()
        list_phase.prop = [galaxy_model]

        # noinspection PyUnresolvedReferences
        assert getattr(list_phase.prop, "0") == galaxy_model

    def test_mix(self, list_phase):
        objects = dict(one=GalaxyModel(), two=Galaxy())

        list_phase.prop = objects

        list_phase.prop.one = Galaxy()

        assert len(list_phase.model.prop) == 2

    def test_named_attributes_in_model(self, list_phase):
        galaxy_model = GalaxyModel(model_redshift=True)
        list_phase.prop = dict(one=galaxy_model)

        assert list_phase.model.prior_count == 1
        assert list_phase.model.prop.one == galaxy_model

        instance = list_phase.model.instance_from_prior_medians()

        assert instance.prop.one is not None
        assert len(instance.prop) == 1

    def test_named_attributes_in_model_override(self, list_phase):
        list_phase.prop = dict(one=GalaxyModel())

        assert list_phase.model.prior_count == 0

        galaxy_model = GalaxyModel(model_redshift=True)

        list_phase.prop.one = galaxy_model

        assert list_phase.model.prior_count == 1
        # assert list_phase.model.one == galaxy_model

        instance = list_phase.model.instance_from_prior_medians()

        # assert instance.one is not None
        assert len(instance.prop) == 1

    def test_named_attributes_in_instance(self, list_phase):
        galaxy = Galaxy()
        list_phase.prop = dict(one=galaxy)

        assert list_phase.model.prior_count == 0
        assert list_phase.model.prop.one == galaxy

    # def test_singular_model_info(self, list_phase):
    #     galaxy_model = test_autoarray.mock.GalaxyModel(model_redshift=True)
    #     list_phase.prop = dict(one=galaxy_model)
    #
    #     assert list_phase.model.prop.one == galaxy_model
    #     assert len(galaxy_model.flat_prior_model_tuples) == 1
    #     assert len(galaxy_model.prior_tuples) == 1
    #
    #     assert len(list_phase.model.flat_prior_model_tuples) == 1
    #
    #     print(list_phase.model.info)
    #
    #     assert len(list_phase.model.info.split('\n')) == 7

    def test_shared_priors(self, list_phase):
        list_phase.prop = dict(
            one=GalaxyModel(model_redshift=True), two=GalaxyModel(model_redshift=True)
        )

        assert list_phase.model.prior_count == 2

        # noinspection PyUnresolvedReferences
        list_phase.prop.one.redshift = list_phase.prop.two.redshift

        assert list_phase.model.prior_count == 1

    def test_hasattr(self, list_phase):
        list_phase.prop = dict()

        assert not hasattr(list_phase.prop, "one")
        list_phase.prop = dict(one=GalaxyModel(model_redshift=True))

        assert hasattr(list_phase.prop, "one")

    def test_position_not_a_prior(self, list_phase):
        list_phase.prop = [af.PriorModel(Galaxy)]

        assert list_phase.model.prior_count == 1
        assert "redshift" == list_phase.model.prior_tuples_ordered_by_id[0][0]

        prior_model = af.PriorModel(Galaxy)
        prior_model.phase_property_position = 0

        print(prior_model.instance_tuples)
        assert len(prior_model.instance_tuples) == 0
