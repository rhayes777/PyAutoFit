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
