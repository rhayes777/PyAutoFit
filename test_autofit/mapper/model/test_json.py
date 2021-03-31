import pytest

import autofit as af
from autofit.mock.mock import Gaussian


@pytest.fixture(
    name="model_dict"
)
def make_model_dict():
    return {
        "type": "model",
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
        "type": "instance",
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


class TestFromDict:
    def test_model_from_dict(
            self,
            model_dict
    ):
        model = af.Model.from_dict(
            model_dict
        )
        assert model.cls == Gaussian


class TestToDict:
    def test_model_priors(
            self,
            model,
            model_dict
    ):
        assert model.dict == model_dict

    def test_model_floats(
            self,
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
            self,
            model,
            model_dict
    ):
        collection = af.Collection(
            gaussian=model
        )
        assert collection.dict == {
            "gaussian": model_dict,
            "type": "collection"
        }

    def test_collection_instance(
            self,
            instance_dict
    ):
        collection = af.Collection(
            gaussian=Gaussian()
        )
        assert collection.dict == {
            "gaussian": instance_dict,
            "type": "collection"
        }
