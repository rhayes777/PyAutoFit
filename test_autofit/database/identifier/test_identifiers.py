import numpy as np
import pytest

import autofit as af
from autofit import conf
from autofit.mapper.model_object import Identifier
from autofit.mock.mock import Gaussian


def set_version(version):
    conf.instance[
        "general"
    ][
        "output"
    ][
        "identifier_version"
    ] = version


@pytest.fixture(
    autouse=True
)
def set_high_version():
    set_version(99)


def test_identifier_version():
    set_version(1)
    identifier = Identifier(Gaussian())

    set_version(2)
    assert identifier != Identifier(Gaussian())

    assert identifier == Identifier(
        Gaussian(),
        version=1
    )


def test_unique_tag_is_used():
    identifier = af.DynestyStatic(
        "name",
        unique_tag="tag"
    ).paths._identifier

    assert "tag" in identifier.hash_list


def test_class_path():
    identifier = Identifier(
        Class,
        version=3
    )
    string, = identifier.hash_list
    assert "test_autofit.database.identifier.test_identifiers.Class" in string


class Class:
    __doc__ = "hello"

    def __init__(self, one=1, two=2, three=3):
        self.one = one
        self.two = two
        self.three = three
        self.four = None

    __identifier_fields__ = ("one", "two")

    def __eq__(self, other):
        return self.one == other.one


class ExcludeClass:
    def __init__(self, one=1, two=2, three=3):
        self.one = one
        self.two = two
        self.three = three

    __exclude_identifier_fields__ = ("three",)


class AttributeClass:
    def __init__(self):
        self.attribute = None


def test_exclude_identifier_fields():
    other = ExcludeClass(
        three=4
    )
    assert Identifier(
        other
    ) == Identifier(
        ExcludeClass()
    )

    other.__exclude_identifier_fields__ = tuple()

    assert Identifier(
        other
    ) != Identifier(
        ExcludeClass()
    )


def test_numpy_array():
    identifier = Identifier(np.array([0]))
    assert identifier.hash_list == []


def test_hash_list():
    identifier = Identifier(Class())
    assert identifier.hash_list == [
        "Class", "one", "1", "two", "2"
    ]


def test_constructor_only():
    attribute = AttributeClass()
    attribute.attribute = 1

    assert Identifier(
        AttributeClass()
    ) == Identifier(
        attribute
    )


def test_exclude_does_no_effect_constructor():
    attribute = AttributeClass()
    attribute.__exclude_identifier_fields__ = tuple()
    attribute.attribute = 1

    assert Identifier(
        AttributeClass()
    ) == Identifier(
        attribute
    )


class PrivateClass:
    def __init__(self, argument):
        self._argument = argument


def test_private_not_included():
    instance = PrivateClass(
        argument="one"
    )
    identifier = str(Identifier(instance))

    instance._argument = "two"
    assert Identifier(instance) == identifier


def test_missing_field():
    instance = Class()
    instance.__identifier_fields__ = ("five",)

    with pytest.raises(
            AssertionError
    ):
        Identifier(
            instance
        )


def test_change_class():
    gaussian_0 = af.Model(
        af.Gaussian,
        intensity=af.UniformPrior(
            lower_limit=1e-6,
            upper_limit=1e6
        )
    )
    gaussian_1 = af.Model(
        af.Gaussian,
        intensity=af.LogUniformPrior(
            lower_limit=1e-6,
            upper_limit=1e6
        )
    )

    assert Identifier(gaussian_0) != Identifier(gaussian_1)


def test_tiny_change():
    # noinspection PyTypeChecker
    instance = Class(
        one=1.0
    )
    identifier = str(Identifier(instance))

    instance.one += 1e-9
    print(instance.one)

    assert identifier == Identifier(instance)


def test_infinity():
    # noinspection PyTypeChecker
    instance = Class(
        one=float("inf")
    )
    str(Identifier(instance))


def test_identifier_fields():
    other = Class(three=4)
    assert Identifier(
        Class()
    ) == Identifier(
        other
    )

    other.__identifier_fields__ = ("one", "two", "three")
    assert Identifier(
        Class()
    ) != Identifier(
        other
    )


def test_unique_tag():
    search = af.MockSearch()

    search.fit(
        model=af.Collection(),
        analysis=af.mock.mock.MockAnalysis()
    )

    identifier = search.paths.identifier

    search = af.MockSearch(unique_tag="dataset")

    search.fit(
        model=af.Collection(),
        analysis=af.mock.mock.MockAnalysis(),
    )

    assert search.paths.identifier != identifier


def test_prior():
    identifier = af.UniformPrior().identifier
    assert identifier == af.UniformPrior().identifier
    assert identifier != af.UniformPrior(
        lower_limit=0.5
    ).identifier
    assert identifier != af.UniformPrior(
        upper_limit=0.5
    ).identifier


def test_model():
    identifier = af.PriorModel(
        Gaussian,
        centre=af.UniformPrior()
    ).identifier
    assert identifier == af.PriorModel(
        Gaussian,
        centre=af.UniformPrior()
    ).identifier
    assert identifier != af.PriorModel(
        Gaussian,
        centre=af.UniformPrior(
            upper_limit=0.5
        )
    ).identifier


def test_collection():
    identifier = af.CollectionPriorModel(
        gaussian=af.PriorModel(
            Gaussian,
            centre=af.UniformPrior()
        )
    ).identifier
    assert identifier == af.CollectionPriorModel(
        gaussian=af.PriorModel(
            Gaussian,
            centre=af.UniformPrior()
        )
    ).identifier
    assert identifier != af.CollectionPriorModel(
        gaussian=af.PriorModel(
            Gaussian,
            centre=af.UniformPrior(
                upper_limit=0.5
            )
        )
    ).identifier


def test_instance():
    identifier = af.CollectionPriorModel(
        gaussian=Gaussian()
    ).identifier
    assert identifier == af.CollectionPriorModel(
        gaussian=Gaussian()
    ).identifier
    assert identifier != af.CollectionPriorModel(
        gaussian=Gaussian(
            centre=0.5
        )
    ).identifier
