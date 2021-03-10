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


def test_both():
    collection = af.CollectionPriorModel(
        gaussian=af.PriorModel(
            m.Gaussian
        ),
        gaussian_2=m.Gaussian()
    )

    assert collection.has_model(
        m.Gaussian
    ) is True
    assert collection.has_instance(
        m.Gaussian
    ) is True


def test_embedded():
    collection = af.CollectionPriorModel(
        gaussian=af.PriorModel(
            m.Gaussian,
            centre=m.Gaussian()
        ),
    )

    assert collection.has_model(
        m.Gaussian
    ) is True
    assert collection.has_instance(
        m.Gaussian
    ) is True


def test_is_only_model():
    collection = af.CollectionPriorModel(
        gaussian=af.PriorModel(
            m.Gaussian
        ),
        gaussian_2=af.PriorModel(
            m.Gaussian
        )
    )

    assert collection.is_only_model(
        m.Gaussian
    ) is True

    collection.other = af.PriorModel(
        m.MockClassx2
    )

    assert collection.is_only_model(
        m.Gaussian
    ) is False
