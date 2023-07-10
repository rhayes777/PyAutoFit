import pytest
import autofit as af


@pytest.fixture(autouse=True)
def reset_priors_by_id():
    af.Prior._priors_by_id = {}


@pytest.fixture(name="model_dict")
def make_model_dict():
    return {
        "type": "model",
        "class_path": "autofit.example.model.Gaussian",
        "centre": {
            "lower_limit": 0.0,
            "type": "Uniform",
            "upper_limit": 2.0,
        },
        "normalization": {
            "lower_limit": 0.0,
            "type": "Uniform",
            "upper_limit": 1.0,
        },
        "sigma": {
            "lower_limit": 0.0,
            "type": "Uniform",
            "upper_limit": 1.0,
        },
    }


@pytest.fixture(name="instance_dict")
def make_instance_dict():
    return {
        "type": "instance",
        "class_path": "autofit.example.model.Gaussian",
        "centre": 0.0,
        "normalization": 0.1,
        "sigma": 0.01,
    }
