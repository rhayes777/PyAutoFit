import pytest

import autofit as af
from autoconf.exc import ConfigException
from autofit.mapper.model_object import Identifier


class SomeWeirdClass:
    def __init__(self, argument):
        self.argument = argument


def test_config_error():
    model = af.Model(
        SomeWeirdClass
    )

    with pytest.raises(ConfigException):
        print(Identifier([
            model
        ]))


def test_mapper_from_prior_arguments_simple_collection():
    old = af.UniformPrior()
    new = af.UniformPrior()
    collection = af.Collection(
        value=old
    )
    collection = collection.mapper_from_prior_arguments({
        old: new
    })

    assert collection.value == new


def test_direct_instances_only():
    child = af.Model(
        af.Gaussian,
        centre=0.0,
        intensity=0.1,
        sigma=0.01,
    )
    child.constant = 1.0

    model = af.Model(
        af.Gaussian,
        centre=child,
        intensity=0.1,
        sigma=0.01,
    )

    new_model = model.gaussian_prior_model_for_arguments({})
    assert not hasattr(new_model, "constant")


def test_function_from_instance():
    assert af.PriorModel.from_instance(
        test_function_from_instance
    ) is test_function_from_instance
