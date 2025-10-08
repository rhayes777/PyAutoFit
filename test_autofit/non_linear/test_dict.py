import pytest

import autofit as af
from autoconf.dictable import to_dict, from_dict


@pytest.fixture(name="dynesty_dict")
def make_dynesty_dict():
    return {
        "arguments": {
            "bootstrap": None,
            "bound": "multi",
            "enlarge": None,
            "facc": 0.5,
            "fmove": 0.9,
            "initial_values": {"arguments": {}, "type": "dict"},
            "initializer": {
                "arguments": {},
                "class_path": "autofit.non_linear.initializer.InitializerPrior",
                "type": "instance",
            },
            "inplace": False,
            "iterations_per_update": 500,
            "max_move": 100,
            "name": "",
            "nlive": 150,
            "number_of_cores": 1,
            "path_prefix": None,
            "paths": {
                "arguments": {},
                "class_path": "autofit.non_linear.paths.null.NullPaths",
                "type": "instance",
            },
            "sample": "auto",
            "slices": 5,
            "unique_tag": None,
            "walks": 5,
        },
        "class_path": "autofit.non_linear.search.nest.dynesty.search.static.DynestyStatic",
        "type": "instance",
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
