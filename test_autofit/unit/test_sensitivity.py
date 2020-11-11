import numpy as np
import pytest

import autofit as af
from autofit import sensitivity as s
from autofit.mock.mock import Gaussian


@pytest.fixture(
    name="perturbation_model"
)
def make_perturbation_model():
    return af.PriorModel(
        Gaussian
    )


@pytest.fixture(
    name="sensitivity"
)
def make_sensitivity(perturbation_model):
    return s.Sensitivity(
        instance=Gaussian(),
        model=af.PriorModel(Gaussian),
        search=af.MockSearch(),
        perturbation_model=perturbation_model,
        image_function=image_function,
        step_size=0.5,
        analysis_class=Analysis
    )


x = np.array(range(10))


def image_function(instance):
    return instance(x) + instance.perturbation(x)


class Analysis(af.Analysis):
    def __init__(self, data):
        self.data = data

    def log_likelihood_function(self, instance):
        return -1


def test_lists(sensitivity):
    assert len(list(sensitivity.perturbation_instances)) == 8


def test_sensitivity(sensitivity):
    results = sensitivity.run()
    assert len(results) == 8


def test_labels(sensitivity):
    labels = list(sensitivity.labels)
    assert labels == [
        'centre_0.25_intensity_0.25_sigma_0.25',
        'centre_0.25_intensity_0.25_sigma_0.75',
        'centre_0.25_intensity_0.75_sigma_0.25',
        'centre_0.25_intensity_0.75_sigma_0.75',
        'centre_0.75_intensity_0.25_sigma_0.25',
        'centre_0.75_intensity_0.25_sigma_0.75',
        'centre_0.75_intensity_0.75_sigma_0.25',
        'centre_0.75_intensity_0.75_sigma_0.75'
    ]


def test_searches(sensitivity):
    assert len(list(sensitivity.searches)) == 8


def test_job(perturbation_model):
    instance = Gaussian()
    instance.perturbation = Gaussian()
    image = image_function(instance)
    job = s.Job(
        model=af.PriorModel(Gaussian),
        perturbation_model=af.PriorModel(Gaussian),
        analysis_class=Analysis,
        search=af.MockSearch(),
        image=image
    )
    result = job.perform()
    assert isinstance(
        result,
        s.JobResult
    )
