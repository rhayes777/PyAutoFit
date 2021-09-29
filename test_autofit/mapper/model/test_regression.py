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
