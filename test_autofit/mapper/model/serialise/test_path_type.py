import autofit as af

import pytest


@pytest.fixture(name="reference")
def make_reference():
    return {
        "gaussian": "autofit.example.model.Gaussian",
        "collection.gaussian": "autofit.example.model.Gaussian",
    }


@pytest.fixture(name="root_reference")
def make_root_reference():
    return {
        "": "autofit.example.model.Gaussian",
    }


def test_reference_model(model_dict, reference):
    model_dict.pop("class_path")
    with_path = {"gaussian": model_dict, "type": "collection"}
    model = af.AbstractModel.from_dict(
        with_path,
        reference=reference,
    )

    assert model.gaussian.cls is af.Gaussian


def test_root_reference(model_dict, root_reference):
    model_dict.pop("class_path")
    model = af.AbstractModel.from_dict(
        model_dict,
        reference=root_reference,
    )

    assert model.cls is af.Gaussian


def test_instance(instance_dict, reference):
    instance_dict.pop("class_path")

    with_path = {"gaussian": instance_dict, "type": "collection"}
    instance = af.AbstractModel.from_dict(
        with_path,
        reference=reference,
    )

    assert isinstance(instance.gaussian, af.Gaussian)


def test_root_instance(instance_dict, root_reference):
    instance_dict.pop("class_path")

    instance = af.AbstractModel.from_dict(
        instance_dict,
        reference=root_reference,
    )

    assert isinstance(instance, af.Gaussian)


def test_deep_reference(instance_dict, reference):
    instance_dict.pop("class_path")

    with_path = {
        "collection": {
            "gaussian": instance_dict,
            "type": "collection",
        },
        "type": "collection",
    }
    instance = af.AbstractModel.from_dict(
        with_path,
        reference=reference,
    )

    assert isinstance(instance.collection.gaussian, af.Gaussian)
