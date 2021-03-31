import pytest

import autofit as af
from autofit.mock.mock import Gaussian


@pytest.fixture(
    name="model_dict"
)
def make_model_dict():
    return {
        "class_path": "autofit.mock.mock.Gaussian",
        "centre": {'lower_limit': 0.0, 'type': 'Uniform', 'upper_limit': 1.0},
        "intensity": {'lower_limit': 0.0, 'type': 'Uniform', 'upper_limit': 1.0},
        "sigma": {'lower_limit': 0.0, 'type': 'Uniform', 'upper_limit': 1.0},
    }


@pytest.fixture(
    name="instance_dict"
)
def make_instance_dict():
    return {
        "class_path": "autofit.mock.mock.Gaussian",
        "centre": 0.0,
        "intensity": 0.1,
        "sigma": 0.01
    }


@pytest.fixture(
    name="model"
)
def make_model():
    return af.Model(
        Gaussian
    )


def test_model_priors_to_dict(
        model,
        model_dict
):
    assert model.dict == model_dict


def test_model_floats_to_dict(
        instance_dict
):
    model = af.Model(
        Gaussian,
        centre=0.0,
        intensity=0.1,
        sigma=0.01
    )

    assert model.dict == instance_dict


def test_collection(
        model,
        model_dict
):
    collection = af.Collection(
        gaussian=model
    )
    assert collection.dict == {
        "gaussian": model_dict
    }


def test_collection_instance(
        instance_dict
):
    collection = af.Collection(
        gaussian=Gaussian()
    )
    assert collection.dict == {
        "gaussian": instance_dict
    }
