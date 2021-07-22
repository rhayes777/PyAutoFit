import pytest

import autofit as af


@pytest.fixture(
    name="model"
)
def make_model():
    return af.Collection(
        gaussian=af.Model(
            af.Gaussian
        )
    )


class TestInstanceFromPathArguments:
    def test(
            self,
            model
    ):
        instance = model.instance_from_path_arguments({
            ("gaussian", "centre"): 0.1,
            ("gaussian", "intensity"): 0.2,
            ("gaussian", "sigma"): 0.3
        })
        assert instance.gaussian.centre == 0.1
        assert instance.gaussian.intensity == 0.2
        assert instance.gaussian.sigma == 0.3

    def test_prior_linking(
            self,
            model
    ):
        model.gaussian.centre = model.gaussian.intensity
        instance = model.instance_from_path_arguments({
            ("gaussian", "centre",): 0.1,
            ("gaussian", "sigma",): 0.3
        })
        assert instance.gaussian.centre == 0.1
        assert instance.gaussian.intensity == 0.1
        assert instance.gaussian.sigma == 0.3

        instance = model.instance_from_path_arguments({
            ("gaussian", "intensity",): 0.2,
            ("gaussian", "sigma",): 0.3
        })
        assert instance.gaussian.centre == 0.2
        assert instance.gaussian.intensity == 0.2
        assert instance.gaussian.sigma == 0.3


def test_instance_from_prior_names(model):
    instance = model.instance_from_prior_name_arguments({
        "gaussian_centre": 0.1,
        "gaussian_intensity": 0.2,
        "gaussian_sigma": 0.3
    })
    assert instance.gaussian.centre == 0.1
    assert instance.gaussian.intensity == 0.2
    assert instance.gaussian.sigma == 0.3
