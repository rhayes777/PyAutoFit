import pytest
import autofit as af


@pytest.fixture(name="model_instance")
def make_model_instance():
    return af.ModelInstance(
        dict(
            t=1.0,
            gaussian=af.Gaussian(centre=0.0, normalization=1.0, sigma=-1.0),
        )
    )


@pytest.fixture(name="instances")
def make_instances(model_instance):
    return [
        model_instance,
        af.ModelInstance(
            dict(
                t=2.0,
                gaussian=af.Gaussian(centre=1.0, normalization=2.0, sigma=-2.0),
            )
        ),
    ]
