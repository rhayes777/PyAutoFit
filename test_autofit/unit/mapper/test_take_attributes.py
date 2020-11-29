import pytest

import autofit as af
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

    assert target_gaussian.centre is prior


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

    assert target.gaussian.centre is prior


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

    assert target[0].centre is prior


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
