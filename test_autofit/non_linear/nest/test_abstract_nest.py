from os import path

import pytest

import autofit as af
from autofit.mock.mock import MockClassx4
from autofit.non_linear.samples import NestSamples, Sample

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(name="samples")
def make_samples():
    model = af.ModelMapper(mock_class_1=MockClassx4)

    parameters = [
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [21.0, 22.0, 23.0, 24.0],
        [0.0, 1.0, 2.0, 3.0],
    ]

    return NestSamples(
        model=model,
        samples=Sample.from_lists(
            model=model,
            parameters=parameters,
            log_likelihoods=[1.0, 2.0, 3.0, 10.0, 5.0],
            log_priors=[0.0, 0.0, 0.0, 0.0, 0.0],
            weights=[1.0, 1.0, 1.0, 1.0, 1.0],
        ),
        total_samples=500,
        log_evidence=2,
        unconverged_sample_size=300,
        time=4,
        number_live_points=5,
    )



