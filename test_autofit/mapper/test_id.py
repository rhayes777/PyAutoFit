import pytest

import autofit as af


@pytest.fixture(
    name="model"
)
def make_model():
    return af.Model(
        af.Gaussian
    )


def test_instance_from_path_arguments(
        model
):
    instance = model.instance_from_path_arguments({
        ("centre",): 1,
        ("intensity",): 2,
        ("sigma",): 3
    })
    assert instance.centre == 1
    assert instance.intensity == 2
    assert instance.sigma == 3


def test_prior_linking(
        model
):
    model.centre = model.intensity
    instance = model.instance_from_path_arguments({
        ("centre",): 1,
        ("sigma",): 3
    })
    assert instance.centre == 1
    assert instance.intensity == 1
    assert instance.sigma == 3

    instance = model.instance_from_path_arguments({
        ("intensity",): 2,
        ("sigma",): 3
    })
    assert instance.centre == 2
    assert instance.intensity == 2
    assert instance.sigma == 3
