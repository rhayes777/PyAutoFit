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
