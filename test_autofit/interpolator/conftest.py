import pytest

import autofit as af
from autofit.interpolator import CovarianceInterpolator


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


@pytest.fixture(name="interpolator")
def make_interpolator():
    samples_list = [
        af.SamplesPDF(
            model=af.Collection(
                t=value,
                gaussian=af.Model(
                    af.Gaussian,
                    centre=af.GaussianPrior(mean=1.0, sigma=1.0),
                    normalization=af.GaussianPrior(mean=1.0, sigma=1.0),
                    sigma=af.GaussianPrior(mean=1.0, sigma=1.0),
                ),
            ),
            sample_list=[
                af.Sample(
                    log_likelihood=-i,
                    log_prior=1.0,
                    weight=1.0,
                    kwargs={
                        ("gaussian", "centre"): value + i,
                        ("gaussian", "normalization"): value + i**2,
                        ("gaussian", "sigma"): value + i**3,
                    },
                )
                for i in range(3)
            ],
        )
        for value in range(3)
    ]
    return CovarianceInterpolator(
        samples_list,
    )
