import pytest

import autofit as af
from autofit.mapper.prior.prior import TuplePrior
from autofit.mock import mock as m


@pytest.fixture(
    name="target_gaussian"
)
def make_target_gaussian():
    return af.PriorModel(
        m.Gaussian
    )


@pytest.fixture(
    name="prior"
)
def make_prior():
    return af.UniformPrior()


@pytest.fixture(
    name="source_gaussian"
)
def make_source_gaussian(prior):
    return af.PriorModel(
        m.Gaussian,
        centre=prior
    )


def test_simple(
        source_gaussian,
        target_gaussian,
        prior
):
    target_gaussian.take_attributes(
        source_gaussian
    )

    assert target_gaussian.centre == prior


def test_assertions(
        source_gaussian,
        target_gaussian
):
    target_gaussian.add_assertion(
        target_gaussian.centre <= target_gaussian.intensity
    )

    with pytest.raises(AssertionError):
        target_gaussian.take_attributes(
            source_gaussian
        )


def test_assertions_collection(
        source_gaussian,
        target_gaussian
):
    target_gaussian.add_assertion(
        target_gaussian.centre <= target_gaussian.intensity
    )

    target_collection = af.Collection(
        gaussian=target_gaussian
    )
    source_collection = af.Collection(
        gaussian=source_gaussian
    )

    with pytest.raises(AssertionError):
        target_collection.take_attributes(
            source_collection
        )


def test_in_collection(
        source_gaussian,
        target_gaussian,
        prior
):
    target = af.CollectionPriorModel(
        gaussian=target_gaussian
    )
    source = af.CollectionPriorModel(
        gaussian=source_gaussian
    )
    target.take_attributes(
        source
    )

    assert target.gaussian.centre == prior


def test_tuple(
        source_gaussian,
        target_gaussian,
        prior
):
    source_gaussian.centre = (prior, 1.0)
    target_gaussian.take_attributes(
        source_gaussian
    )

    assert target_gaussian.centre == (prior, 1.0)


def test_tuple_prior(
        source_gaussian,
        target_gaussian,
        prior
):
    source_gaussian.centre = (prior, 1.0)
    target_gaussian.centre = TuplePrior()
    target_gaussian.take_attributes(
        source_gaussian
    )

    assert target_gaussian.centre == (prior, 1.0)


def test_tuple_in_instance(
        target_gaussian,
        prior
):
    # noinspection PyTypeChecker
    source_gaussian = m.Gaussian(
        centre=(prior, 1.0)
    )
    target_gaussian.take_attributes(
        source_gaussian
    )

    assert target_gaussian.centre == (prior, 1.0)


def test_tuple_in_collection(
        source_gaussian,
        target_gaussian,
        prior
):
    source_gaussian.centre = (prior, 1.0)

    source = af.CollectionPriorModel(
        gaussian=source_gaussian
    )
    target = af.CollectionPriorModel(
        gaussian=target_gaussian
    )

    target.take_attributes(source)
    assert target.gaussian.centre == (prior, 1.0)


def test_tuple_in_instance_in_collection(
        target_gaussian,
        prior
):
    # noinspection PyTypeChecker
    source_gaussian = m.Gaussian(
        centre=(prior, 1.0)
    )

    source = af.CollectionPriorModel(
        gaussian=source_gaussian
    )
    target = af.CollectionPriorModel(
        gaussian=target_gaussian
    )

    target.take_attributes(source)
    assert target.gaussian.centre == (prior, 1.0)


def test_source_is_dict(
        source_gaussian,
        target_gaussian,
        prior
):
    source = dict(
        gaussian=source_gaussian
    )
    target = af.CollectionPriorModel(
        gaussian=target_gaussian
    )
    target.take_attributes(source)

    assert target.gaussian.centre == prior


def test_target_is_dict(
        source_gaussian,
        target_gaussian,
        prior
):
    source = af.CollectionPriorModel(
        collection=af.CollectionPriorModel(
            gaussian=source_gaussian
        )
    )
    target = af.CollectionPriorModel(
        collection=dict(
            gaussian=target_gaussian
        )
    )
    target.take_attributes(source)

    assert target.collection.gaussian.centre == prior


def test_missing_from_source(
        target_gaussian,
        prior
):
    target_gaussian.centre = prior

    target_gaussian.take_attributes(
        af.CollectionPriorModel()
    )
    assert target_gaussian.centre == prior


def test_unlabelled_in_collection(
        source_gaussian,
        target_gaussian,
        prior
):
    target = af.CollectionPriorModel(
        [target_gaussian]
    )
    source = af.CollectionPriorModel(
        [source_gaussian]
    )
    target.take_attributes(
        source
    )

    assert target[0].centre == prior


def test_passing_float(
        source_gaussian,
        target_gaussian
):
    source_gaussian.centre = 2.0
    target_gaussian.take_attributes(
        source_gaussian
    )

    assert target_gaussian.centre == 2.0


def test_missing_from_origin(
        target_gaussian
):
    target_gaussian.take_attributes(
        af.CollectionPriorModel()
    )
