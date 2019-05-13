import os

import pytest

import autofit.mapper.prior_model
from autofit import conf
from autofit import mock
from autofit.mapper import model_mapper as mm
from autofit.optimize import non_linear
from autofit.tools import phase as ph
from autofit.tools import phase_property

directory = os.path.dirname(os.path.realpath(__file__))

conf.instance = conf.Config("{}/../../workspace/config".format(directory),
                            "{}/../../workspace/output/".format(directory))


class NLO(non_linear.NonLinearOptimizer):
    def fit(self, analysis):
        class Fitness(object):
            def __init__(self, instance_from_physical_vector, constant):
                self.result = None
                self.instance_from_physical_vector = instance_from_physical_vector
                self.constant = constant

            def __call__(self, vector):
                instance = self.instance_from_physical_vector(vector)
                for key, value in self.constant.__dict__.items():
                    setattr(instance, key, value)

                likelihood = analysis.fit(instance)
                self.result = non_linear.Result(instance, likelihood)

                # Return Chi squared
                return -2 * likelihood

        fitness_function = Fitness(self.variable.instance_from_physical_vector, self.constant)
        fitness_function(self.variable.prior_count * [0.5])

        return fitness_function.result


@pytest.fixture(name='phase')
def make_phase():
    class MyPhase(ph.AbstractPhase):
        prop = phase_property.PhaseProperty("prop")

    return MyPhase(phase_name='', optimizer_class=NLO)


@pytest.fixture(name='list_phase')
def make_list_phase():
    class MyPhase(ph.AbstractPhase):
        prop = phase_property.PhaseProperty("prop")

    return MyPhase(phase_name='', optimizer_class=NLO)


class TestPhasePropertyList(object):
    # def test_constants(self, list_phase):
    #     objects = [mock.Galaxy(), mock.Galaxy()]
    #
    #     list_phase.prop = objects
    #
    #     assert list_phase.constant.prop == objects
    #     assert list_phase.prop == objects

    def test_classes(self, list_phase):
        objects = [mock.GalaxyModel(), mock.GalaxyModel()]

        list_phase.prop = objects

        assert list_phase.variable.prop == objects
        assert list_phase.prop == objects

    def test_abstract_prior_models(self, list_phase):
        objects = [autofit.mapper.prior_model.AbstractPriorModel(), autofit.mapper.prior_model.AbstractPriorModel()]

        list_phase.prop = objects

        assert list_phase.variable.prop == objects
        assert list_phase.prop == objects

    def test_mix(self, list_phase):
        objects = [mock.GalaxyModel(), mock.Galaxy()]

        list_phase.prop = objects

        assert list_phase.variable.prop == objects
        assert list_phase.prop == objects

    def test_set_item(self, list_phase):
        galaxy_prior_0 = mock.GalaxyModel()
        objects = [galaxy_prior_0, mock.Galaxy()]

        list_phase.prop = objects

        galaxy_prior_1 = mock.GalaxyModel()
        list_phase.prop[1] = galaxy_prior_1

        assert list_phase.variable.prop == [galaxy_prior_0, galaxy_prior_1]

        galaxy = mock.Galaxy()

        list_phase.prop[0] = galaxy
        assert list_phase.prop == [galaxy, galaxy_prior_1]


class TestPhasePropertyCollectionAttributes(object):
    def test_set_list_as_dict(self, list_phase):
        galaxy_model = mock.GalaxyModel()
        list_phase.prop = dict(one=galaxy_model)

        assert len(list_phase.prop) == 1
        # noinspection PyUnresolvedReferences
        assert list_phase.prop.one == galaxy_model

    def test_override_property(self, list_phase):
        galaxy_model = mock.GalaxyModel()

        list_phase.prop = dict(one=mock.GalaxyModel())

        list_phase.prop.one = galaxy_model

        assert len(list_phase.prop) == 1
        assert list_phase.prop.one == galaxy_model

    def test_named_list_items(self, list_phase):
        galaxy_model = mock.GalaxyModel()
        list_phase.prop = [galaxy_model]

        # noinspection PyUnresolvedReferences
        assert getattr(list_phase.prop, "0") == galaxy_model

    def test_mix(self, list_phase):
        objects = dict(one=mock.GalaxyModel(), two=mock.Galaxy())

        list_phase.prop = objects

        list_phase.prop.one = mock.Galaxy()

        assert len(list_phase.variable.prop) == 2

    def test_named_attributes_in_variable(self, list_phase):
        galaxy_model = mock.GalaxyModel(variable_redshift=True)
        list_phase.prop = dict(one=galaxy_model)

        assert list_phase.variable.prior_count == 1
        assert list_phase.variable.prop.one == galaxy_model

        instance = list_phase.variable.instance_from_prior_medians()

        assert instance.prop.one is not None
        assert len(instance.prop) == 1

    def test_named_attributes_in_variable_override(self, list_phase):
        list_phase.prop = dict(one=mock.GalaxyModel())

        assert list_phase.variable.prior_count == 0

        galaxy_model = mock.GalaxyModel(variable_redshift=True)

        list_phase.prop.one = galaxy_model

        assert list_phase.variable.prior_count == 1
        # assert list_phase.variable.one == galaxy_model

        instance = list_phase.variable.instance_from_prior_medians()

        # assert instance.one is not None
        assert len(instance.prop) == 1

    def test_named_attributes_in_constant(self, list_phase):
        galaxy = mock.Galaxy()
        list_phase.prop = dict(one=galaxy)

        assert list_phase.variable.prior_count == 0
        assert list_phase.variable.prop.one == galaxy

    def test_singular_model_info(self, list_phase):
        galaxy_model = mock.GalaxyModel(variable_redshift=True)
        list_phase.prop = dict(one=galaxy_model)

        assert list_phase.variable.prop.one == galaxy_model
        assert len(galaxy_model.flat_prior_model_tuples) == 1
        assert len(galaxy_model.prior_tuples) == 1

        assert len(list_phase.variable.flat_prior_model_tuples) == 1

        print(list_phase.variable.info)

        assert len(list_phase.variable.info.split('\n')) == 7

    def test_shared_priors(self, list_phase):
        list_phase.prop = dict(one=mock.GalaxyModel(variable_redshift=True),
                               two=mock.GalaxyModel(variable_redshift=True))

        assert list_phase.variable.prior_count == 2

        # noinspection PyUnresolvedReferences
        list_phase.prop.one.redshift = list_phase.prop.two.redshift

        assert list_phase.variable.prior_count == 1

    def test_hasattr(self, list_phase):
        list_phase.prop = dict()

        assert not hasattr(list_phase.prop, "one")
        list_phase.prop = dict(one=mock.GalaxyModel(variable_redshift=True))

        assert hasattr(list_phase.prop, "one")

    def test_position_not_a_prior(self, list_phase):
        list_phase.prop = [autofit.mapper.prior_model.PriorModel(mock.Galaxy)]

        assert list_phase.variable.prior_count == 1
        assert "redshift" == list_phase.variable.prior_tuples_ordered_by_id[0][0]

        prior_model = autofit.mapper.prior_model.PriorModel(mock.Galaxy)
        prior_model.phase_property_position = 0

        assert len(prior_model.constant_tuples) == 0
