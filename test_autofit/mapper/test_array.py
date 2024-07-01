import pytest

import autofit as af


@pytest.fixture
def array():
    return af.Array(
        shape=(2, 2),
        prior=af.GaussianPrior(mean=0.0, sigma=1.0),
    )


@pytest.fixture
def array_3d():
    return af.Array(
        shape=(2, 2, 2),
        prior=af.GaussianPrior(mean=0.0, sigma=1.0),
    )


def test_prior_count(array):
    assert array.prior_count == 4


def test_prior_count_3d(array_3d):
    assert array_3d.prior_count == 8


def test_instance(array):
    instance = array.instance_from_prior_medians()
    assert (instance == [[0.0, 0.0], [0.0, 0.0]]).all()


def test_instance_3d(array_3d):
    instance = array_3d.instance_from_prior_medians()
    assert (
        instance
        == [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]
    ).all()


def test_modify_prior(array):
    array[0, 0] = 1.0
    assert array.prior_count == 3
