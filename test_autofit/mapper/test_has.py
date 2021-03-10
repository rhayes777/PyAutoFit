import autofit as af
from autofit.mock import mock as m


def test_instance():
    collection = af.CollectionPriorModel(
        gaussian=m.Gaussian()
    )

    assert collection.has_instance(
        m.Gaussian
    ) is True
    assert collection.has_model(
        m.Gaussian
    ) is False


def test_model():
    collection = af.CollectionPriorModel(
        gaussian=af.PriorModel(
            m.Gaussian
        )
    )

    assert collection.has_model(
        m.Gaussian
    ) is True
    assert collection.has_instance(
        m.Gaussian
    ) is False
