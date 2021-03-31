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
        "centre": {'lower_limit': 0.0, 'type': 'Uniform', 'upper_limit': 2.0},
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
    name="collection_dict"
)
def make_collection_dict(
        model_dict
):
    return {
        "gaussian": model_dict,
        "type": "collection"
    }


@pytest.fixture(
    name="model"
)
def make_model():
    return af.Model(
        Gaussian,
        centre=af.UniformPrior(
            upper_limit=2.0
        )
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
        assert model.prior_count == 3
        assert model.centre.upper_limit == 2.0

    def test_instance_from_dict(
            self,
            instance_dict
    ):
        instance = af.Model.from_dict(
            instance_dict
        )
        assert isinstance(
            instance,
            Gaussian
        )
        assert instance.centre == 0.0
        assert instance.intensity == 0.1
        assert instance.sigma == 0.01

    def test_collection_from_dict(
            self,
            collection_dict
    ):
        collection = af.Model.from_dict(
            collection_dict
        )
        assert isinstance(
            collection,
            af.Collection
        )
        assert len(collection) == 1


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
            collection_dict
    ):
        collection = af.Collection(
            gaussian=model
        )
        assert collection.dict == collection_dict

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
