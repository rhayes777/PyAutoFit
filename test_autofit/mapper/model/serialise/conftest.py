import pytest


@pytest.fixture(name="model_dict")
def make_model_dict():
    return {
        "class_path": "autofit.example.model.Gaussian",
        "type": "model",
        "arguments": {
            "centre": {"lower_limit": 0.0, "upper_limit": 2.0, "type": "Uniform"},
            "normalization": {
                "lower_limit": 0.0,
                "upper_limit": 1.0,
                "type": "Uniform",
            },
            "sigma": {"lower_limit": 0.0, "upper_limit": 1.0, "type": "Uniform"},
        },
    }


@pytest.fixture(name="instance_dict")
def make_instance_dict():
    return {
        "class_path": "autofit.example.model.Gaussian",
        "type": "instance",
        "arguments": {
            "centre": {"type": "Constant", "value": 0.0},
            "normalization": {"type": "Constant", "value": 0.1},
            "sigma": {"type": "Constant", "value": 0.01},
        },
    }
