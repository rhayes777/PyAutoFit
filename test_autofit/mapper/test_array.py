import pytest

import autofit as af
from autoconf.dictable import to_dict


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
    assert (
        array.instance_from_prior_medians()
        == [
            [1.0, 0.0],
            [0.0, 0.0],
        ]
    ).all()


def test_correlation(array):
    array[0, 0] = array[1, 1]
    array[0, 1] = array[1, 0]

    instance = array.random_instance()

    assert instance[0, 0] == instance[1, 1]
    assert instance[0, 1] == instance[1, 0]


@pytest.fixture
def array_dict():
    return {
        "arguments": {
            "indices": {
                "type": "list",
                "values": [
                    {"type": "tuple", "values": [0, 0]},
                    {"type": "tuple", "values": [0, 1]},
                    {"type": "tuple", "values": [1, 0]},
                    {"type": "tuple", "values": [1, 1]},
                ],
            },
            "prior_0_0": {
                "lower_limit": float("-inf"),
                "mean": 0.0,
                "sigma": 1.0,
                "type": "Gaussian",
                "upper_limit": float("inf"),
            },
            "prior_0_1": {
                "lower_limit": float("-inf"),
                "mean": 0.0,
                "sigma": 1.0,
                "type": "Gaussian",
                "upper_limit": float("inf"),
            },
            "prior_1_0": {
                "lower_limit": float("-inf"),
                "mean": 0.0,
                "sigma": 1.0,
                "type": "Gaussian",
                "upper_limit": float("inf"),
            },
            "prior_1_1": {
                "lower_limit": float("-inf"),
                "mean": 0.0,
                "sigma": 1.0,
                "type": "Gaussian",
                "upper_limit": float("inf"),
            },
            "shape": {"type": "tuple", "values": [2, 2]},
        },
        "type": "array",
    }


def test_to_dict(array, array_dict, remove_ids):
    assert remove_ids(to_dict(array)) == array_dict


def test_from_dict(array_dict):
    array = af.AbstractPriorModel.from_dict(array_dict)
    assert array.prior_count == 4
    assert (
        array.instance_from_prior_medians()
        == [
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    ).all()


@pytest.fixture
def array_1d():
    return af.Array(
        shape=(2,),
        prior=af.GaussianPrior(mean=0.0, sigma=1.0),
    )


def test_1d_array(array_1d):
    assert array_1d.prior_count == 2
    assert (array_1d.instance_from_prior_medians() == [0.0, 0.0]).all()
