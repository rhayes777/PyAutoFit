import pytest

import autofit as af
from autoconf.exc import ConfigException
from autofit.mapper.model_object import Identifier


class SomeWeirdClass:
    def __init__(self, argument):
        self.argument = argument


def test_config_error():
    model = af.Model(SomeWeirdClass)

    with pytest.raises(ConfigException):
        print(Identifier([model]))


def test_mapper_from_prior_arguments_simple_collection():
    old = af.UniformPrior()
    new = af.UniformPrior()
    collection = af.Collection(value=old)
    collection = collection.mapper_from_prior_arguments({old: new})

    assert collection.value == new


def test_direct_instances_only():
    child = af.Model(
        af.Gaussian,
        centre=0.0,
        normalization=0.1,
        sigma=0.01,
    )
    child.constant = 1.0

    model = af.Model(
        af.Gaussian,
        centre=child,
        normalization=0.1,
        sigma=0.01,
    )

    new_model = model.gaussian_prior_model_for_arguments({})
    assert not hasattr(new_model, "constant")


def test_function_from_instance():
    assert (
        af.Model.from_instance(test_function_from_instance)
        is test_function_from_instance
    )


def test_as_model_tuples():
    model = af.Model(af.m.MockWithTuple)
    assert isinstance(model.tup.tup_0, af.UniformPrior)
    assert isinstance(model.tup.tup_1, af.UniformPrior)

    instance = model.instance_from_prior_medians()
    assert instance.tup == (0.5, 0.5)

    model = af.AbstractPriorModel.from_instance(instance)
    assert model.tup == (0.5, 0.5)
    assert (
        """tup                                                                             (0.5, 0.5)"""
        in model.info
    )


def test_info_prints_number_of_parameters():
    model = af.Model(af.Gaussian)
    assert "Total Free Parameters" in model.info


def test_set_centre():
    model = af.Model(af.m.MockWithTuple)
    model.tup_0 = 10.0

    instance = model.instance_from_prior_medians()
    assert instance.tup[0] == 10.0

    model = af.Model(af.m.MockWithTuple)
    model.tup.tup_0 = 10.0

    instance = model.instance_from_prior_medians()
    assert instance.tup[0] == 10.0


def test_passing_priors():
    model = af.Model(af.m.MockWithTuple)

    new_model = model.mapper_from_gaussian_tuples(
        [
            (1, 1),
            (1, 1),
        ]
    )
    assert isinstance(new_model.tup_0, af.GaussianPrior)
    assert isinstance(new_model.tup_1, af.GaussianPrior)


def test_passing_fixed():
    model = af.Model(af.m.MockWithTuple)
    model.tup_0 = 0.1
    model.tup_1 = 2.0

    new_model = model.mapper_from_gaussian_tuples([])
    assert new_model.tup_0 == 0.1
    assert new_model.tup_1 == 2.0


def test_independent_ids():
    prior = af.UniformPrior()
    af.ModelInstance()
    assert af.UniformPrior().id == prior.id + 1


@pytest.fixture(name="gaussian")
def make_gaussian():
    return af.Gaussian()


@pytest.fixture(name="instance")
def make_instance(gaussian):
    return af.ModelInstance(dict(ls=[gaussian]))


@pytest.fixture(name="path")
def make_path():
    return "ls", 0


def test_lists(instance, gaussian, path):
    assert instance.path_instance_tuples_for_class(af.Gaussian) == [(path, gaussian)]


def test_replace_positional_path(instance, gaussian, path):
    new = instance.replacing_for_path(path, None)
    assert new.ls[0] is None


def test_ignore_prior_limits():
    model = af.Model(af.Gaussian)
    model.add_assertion(model.centre < 0.0)

    model.instance_from_vector(
        [0.5, 0.5, 0.5],
        ignore_prior_limits=True,
    )
