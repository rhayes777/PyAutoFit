import pytest

import autofit as af
from autofit.example import Exponential


class GaussianChild(af.ex.Gaussian):
    pass


def test_inheritance():
    collection = af.Collection(first=af.Model(GaussianChild), second=GaussianChild())

    assert collection.has_instance(af.ex.Gaussian)
    assert collection.has_model(af.ex.Gaussian)


def test_no_free_parameters():
    collection = af.Collection(
        gaussian=af.Model(af.ex.Gaussian, centre=1.0, normalization=1.0, sigma=1.0,)
    )
    assert collection.prior_count == 0
    assert collection.has_model(af.ex.Gaussian) is False


def test_instance():
    collection = af.Collection(gaussian=af.ex.Gaussian())

    assert collection.has_instance(af.ex.Gaussian) is True
    assert collection.has_model(af.ex.Gaussian) is False


def test_model():
    collection = af.Collection(gaussian=af.Model(af.ex.Gaussian))

    assert collection.has_model(af.ex.Gaussian) is True
    assert collection.has_instance(af.ex.Gaussian) is False


def test_both():
    collection = af.Collection(gaussian=af.Model(af.ex.Gaussian), gaussian_2=af.ex.Gaussian())

    assert collection.has_model(af.ex.Gaussian) is True
    assert collection.has_instance(af.ex.Gaussian) is True


def test_embedded():
    collection = af.Collection(gaussian=af.Model(af.ex.Gaussian, centre=af.ex.Gaussian()),)

    assert collection.has_model(af.ex.Gaussian) is True
    assert collection.has_instance(af.ex.Gaussian) is True


def test_is_only_model():
    collection = af.Collection(
        gaussian=af.Model(af.ex.Gaussian), gaussian_2=af.Model(af.ex.Gaussian)
    )

    assert collection.is_only_model(af.ex.Gaussian) is True

    collection.other = af.Model(af.m.MockClassx2)

    assert collection.is_only_model(af.ex.Gaussian) is False


@pytest.fixture(name="collection")
def make_collection():
    return af.Collection(
        gaussian=af.Model(af.ex.Gaussian), exponential=af.Model(Exponential),
    )


def test_models(collection):
    assert collection.models_with_type(af.ex.Gaussian) == [collection.gaussian]


def test_multiple_types(collection):
    assert collection.models_with_type((af.ex.Gaussian, Exponential)) == [
        collection.gaussian,
        collection.exponential,
    ]


class Galaxy:
    def __init__(self, child):
        self.child = child


def test_instances_with_type():
    model = af.Collection(galaxy=Galaxy(child=af.Model(af.ex.Gaussian)))
    assert model.models_with_type(af.ex.Gaussian) == [model.galaxy.child]


class DelaunayBrightnessImage:
    pass


def test_model_attributes_with_type():
    mesh = af.Model(DelaunayBrightnessImage)
    mesh.pixels = af.UniformPrior(lower_limit=5.0, upper_limit=10.0)
    pixelization = af.Model(af.ex.Gaussian, mesh=mesh)

    assert pixelization.models_with_type(DelaunayBrightnessImage) == [mesh]
