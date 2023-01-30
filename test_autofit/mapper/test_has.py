import pytest

import autofit as af
from autofit.example import Exponential


class GaussianChild(af.Gaussian):
    pass


def test_inheritance():
    collection = af.Collection(first=af.Model(GaussianChild), second=GaussianChild())

    assert collection.has_instance(af.Gaussian)
    assert collection.has_model(af.Gaussian)


def test_no_free_parameters():
    collection = af.Collection(
        gaussian=af.Model(af.Gaussian, centre=1.0, normalization=1.0, sigma=1.0,)
    )
    assert collection.prior_count == 0
    assert collection.has_model(af.Gaussian) is False


def test_instance():
    collection = af.Collection(gaussian=af.Gaussian())

    assert collection.has_instance(af.Gaussian) is True
    assert collection.has_model(af.Gaussian) is False


def test_model():
    collection = af.Collection(gaussian=af.Model(af.Gaussian))

    assert collection.has_model(af.Gaussian) is True
    assert collection.has_instance(af.Gaussian) is False


def test_both():
    collection = af.Collection(gaussian=af.Model(af.Gaussian), gaussian_2=af.Gaussian())

    assert collection.has_model(af.Gaussian) is True
    assert collection.has_instance(af.Gaussian) is True


def test_embedded():
    collection = af.Collection(gaussian=af.Model(af.Gaussian, centre=af.Gaussian()),)

    assert collection.has_model(af.Gaussian) is True
    assert collection.has_instance(af.Gaussian) is True


def test_is_only_model():
    collection = af.Collection(
        gaussian=af.Model(af.Gaussian), gaussian_2=af.Model(af.Gaussian)
    )

    assert collection.is_only_model(af.Gaussian) is True

    collection.other = af.Model(af.m.MockClassx2)

    assert collection.is_only_model(af.Gaussian) is False


@pytest.fixture(name="collection")
def make_collection():
    return af.Collection(
        gaussian=af.Model(af.Gaussian), exponential=af.Model(Exponential),
    )


def test_models(collection):
    assert collection.models_with_type(af.Gaussian) == [collection.gaussian]


def test_multiple_types(collection):
    assert collection.models_with_type((af.Gaussian, Exponential)) == [
        collection.gaussian,
        collection.exponential,
    ]


class Galaxy:
    def __init__(self, child):
        self.child = child


def test_instances_with_type():
    model = af.Collection(galaxy=Galaxy(child=af.Model(af.Gaussian)))
    assert model.models_with_type(af.Gaussian) == [model.galaxy.child]


class DelaunayBrightnessImage:
    pass


def test_model_attributes_with_type():
    mesh = af.Model(DelaunayBrightnessImage)
    mesh.pixels = af.UniformPrior(lower_limit=5.0, upper_limit=10.0)
    pixelization = af.Model(af.Gaussian, mesh=mesh)

    assert pixelization.models_with_type(DelaunayBrightnessImage) == [mesh]
