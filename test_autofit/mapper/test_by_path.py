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


@pytest.fixture(
    name="underscore_model"
)
def make_underscore_model():
    return af.Collection(
        gaussian_component=af.Model(
            af.Gaussian
        )
    )


class TestInstanceFromPriorNames:
    def test(self, model):
        instance = model.instance_from_prior_name_arguments({
            "gaussian_centre": 0.1,
            "gaussian_intensity": 0.2,
            "gaussian_sigma": 0.3
        })
        assert instance.gaussian.centre == 0.1
        assert instance.gaussian.intensity == 0.2
        assert instance.gaussian.sigma == 0.3

    def test_underscored_names(self, underscore_model):
        instance = underscore_model.instance_from_prior_name_arguments({
            "gaussian_component_centre": 0.1,
            "gaussian_component_intensity": 0.2,
            "gaussian_component_sigma": 0.3
        })
        assert instance.gaussian_component.centre == 0.1
        assert instance.gaussian_component.intensity == 0.2
        assert instance.gaussian_component.sigma == 0.3

    def test_prior_linking(self, underscore_model):
        underscore_model.gaussian_component.intensity = (
            underscore_model.gaussian_component.centre
        )
        instance = underscore_model.instance_from_prior_name_arguments({
            "gaussian_component_centre": 0.1,
            "gaussian_component_sigma": 0.3
        })
        assert instance.gaussian_component.centre == 0.1
        assert instance.gaussian_component.intensity == 0.1
        assert instance.gaussian_component.sigma == 0.3

        instance = underscore_model.instance_from_prior_name_arguments({
            "gaussian_component_intensity": 0.2,
            "gaussian_component_sigma": 0.3
        })
        assert instance.gaussian_component.centre == 0.2
        assert instance.gaussian_component.intensity == 0.2
        assert instance.gaussian_component.sigma == 0.3

    def test_path_for_name(self, underscore_model):
        assert underscore_model.path_for_name(
            "gaussian_component_centre"
        ) == (
                   "gaussian_component",
                   "centre"
               )

