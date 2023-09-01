import pytest

import autofit as af
from autofit.tools.util import to_dict, from_dict


@pytest.fixture(name="dynesty_dict")
def make_dynesty_dict():
    return {
        "type": "instance",
        "class_path": "autofit.non_linear.search.nest.dynesty.search.static.DynestyStatic",
        "arguments": {
            "bound": "multi",
            "max_move": 100,
            "nlive": 150,
            "fmove": 0.9,
            "enlarge": None,
            "bootstrap": None,
            "walks": 5,
            "sample": "auto",
            "name": "",
            "slices": 5,
            "facc": 0.5,
            "unique_tag": None,
            "path_prefix": None,
            "iterations_per_update": 500,
            "number_of_cores": 1,
        },
    }


def test_dict(dynesty_dict):
    dynesty = af.DynestyStatic()
    assert to_dict(dynesty) == dynesty_dict


def test_from_dict(dynesty_dict):
    assert isinstance(from_dict(dynesty_dict), af.DynestyStatic)


def test_initializer():
    initializer = af.InitializerBall(lower_limit=0.0, upper_limit=1.0)
    assert to_dict(initializer) == {
        "arguments": {"lower_limit": 0.0, "upper_limit": 1.0},
        "type": "instance",
        "class_path": "autofit.non_linear.initializer.InitializerBall",
    }


class ClassWithType:
    def __init__(self, type_: type):
        self.type_ = type_


@pytest.fixture(name="type_dict")
def make_type_dict():
    return {
        "type": "instance",
        "class_path": "test_autofit.non_linear.test_dict.ClassWithType",
        "arguments": {
            "type_": {
                "type": "type",
                "class_path": "autofit.non_linear.initializer.InitializerBall",
            }
        },
    }


def test_type_to_dict(type_dict):
    cls = ClassWithType(type_=af.InitializerBall)
    assert to_dict(cls) == type_dict


def test_type_from_dict(type_dict):
    assert isinstance(from_dict(type_dict), ClassWithType)
