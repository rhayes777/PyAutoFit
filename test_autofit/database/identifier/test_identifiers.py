import pytest

import autofit as af
from autofit.mapper.model_object import Identifier
from autofit.mock.mock import Gaussian


class Class:
    def __init__(self, one=1, two=2, three=3):
        self.one = one
        self.two = two
        self.three = three
        self.four = None

    __identifier_fields__ = ("one", "two")


class AttributeClass:
    def __init__(self):
        self.attribute = None


def test_constructor_only():
    attribute = AttributeClass()
    attribute.attribute = 1

    assert Identifier(
        AttributeClass()
    ) == Identifier(
        attribute
    )


def test_missing_field():
    instance = Class()
    instance.__identifier_fields__ = ("five",)

    with pytest.raises(
            AssertionError
    ):
        Identifier(
            instance
        )


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
