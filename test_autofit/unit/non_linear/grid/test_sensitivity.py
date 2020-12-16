import numpy as np
import pytest

import autofit as af
from autofit.mock.mock import Gaussian
from autofit.non_linear.grid import sensitivity as s
from autofit.non_linear.grid.simple_grid import GridSearch


@pytest.fixture(name="perturbation_model")
def make_perturbation_model():
    return af.PriorModel(Gaussian)


@pytest.fixture(name="sensitivity")
def make_sensitivity(perturbation_model):
    # noinspection PyTypeChecker
    instance = af.ModelInstance()
    instance.gaussian = Gaussian()
    return s.Sensitivity(
        instance=instance,
        model=af.Collection(
            gaussian=af.PriorModel(Gaussian)
        ),
        perturbation_model=perturbation_model,
        simulate_function=image_function,
        analysis_class=Analysis,
        search=GridSearch(),
        step_size=0.5,
    )


x = np.array(range(10))


def image_function(instance: af.ModelInstance):
    image = instance.gaussian(x)
    if hasattr(instance, "perturbation"):
        image += instance.perturbation(x)
    return image


class Analysis:

    def __init__(self, image: np.array):
        self.image = image

    def log_likelihood_function(self, instance):
        image = image_function(instance)
        return np.mean(np.multiply(-0.5, np.square(np.subtract(self.image, image))))


def test_lists(sensitivity):
    assert len(list(sensitivity._perturbation_instances)) == 8


def test_sensitivity(sensitivity):
    results = sensitivity.run()
    assert len(results) == 8

    for result in results:
        assert result.log_likelihood_difference > 0


def test_tuple_step_size(sensitivity):
    sensitivity.step_size = (0.5, 0.5, 0.25)
    assert len(sensitivity._lists) == 16


def test_labels(sensitivity):
    labels = list(sensitivity._labels)
    assert labels == [
        "centre_0.25_intensity_0.25_sigma_0.25",
        "centre_0.25_intensity_0.25_sigma_0.75",
        "centre_0.25_intensity_0.75_sigma_0.25",
        "centre_0.25_intensity_0.75_sigma_0.75",
        "centre_0.75_intensity_0.25_sigma_0.25",
        "centre_0.75_intensity_0.25_sigma_0.75",
        "centre_0.75_intensity_0.75_sigma_0.25",
        "centre_0.75_intensity_0.75_sigma_0.75",
    ]


def test_searches(sensitivity):
    assert len(list(sensitivity._searches)) == 8


def test_job(perturbation_model):
    instance = af.ModelInstance()
    instance.gaussian = Gaussian()
    instance.perturbation = Gaussian()
    image = image_function(instance)
    # noinspection PyTypeChecker
    job = s.Job(
        model=af.Collection(
            gaussian=af.PriorModel(Gaussian)
        ),
        perturbation_model=af.PriorModel(Gaussian),
        analysis=Analysis(image),
        search=GridSearch(),
    )
    result = job.perform()
    assert isinstance(result, s.JobResult)
    assert isinstance(result.perturbed_result, af.Result)
    assert isinstance(result.result, af.Result)
    assert result.log_likelihood_difference > 0
