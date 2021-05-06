from os import path

import pytest

import autofit as af
from autofit.mock.mock import MockClassx4
from autofit.non_linear.samples import MCMCSamples, Sample
from autofit.non_linear.mcmc.auto_correlations import AutoCorrelations

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(name="samples")
def make_samples():
    model = af.ModelMapper(mock_class_1=MockClassx4)
    print(model.path_priors_tuples)

    parameters = [
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [21.0, 22.0, 23.0, 24.0],
        [0.0, 1.0, 2.0, 3.0],
    ]

    samples = [
        Sample(
            log_likelihood=1.0,
            log_prior=0.0,
            weights=1.0,
            mock_class_1_one=0.0,
            mock_class_1_two=1.0,
            mock_class_1_three=2.0,
            mock_class_1_four=3.0
        ),
        Sample(
            log_likelihood=2.0,
            log_prior=0.0,
            weights=1.0,
            mock_class_1_one=0.0,
            mock_class_1_two=1.0,
            mock_class_1_three=2.0,
            mock_class_1_four=3.0
        ),
        Sample(
            log_likelihood=3.0,
            log_prior=0.0,
            weights=1.0,
            mock_class_1_one=0.0,
            mock_class_1_two=1.0,
            mock_class_1_three=2.0,
            mock_class_1_four=3.0
        ),
        Sample(
            log_likelihood=10.0,
            log_prior=0.0,
            weights=1.0,
            mock_class_1_one=21.0,
            mock_class_1_two=22.0,
            mock_class_1_three=23.0,
            mock_class_1_four=24.0
        ),
        Sample(
            log_likelihood=5.0,
            log_prior=0.0,
            weights=1.0,
            mock_class_1_one=0.0,
            mock_class_1_two=1.0,
            mock_class_1_three=2.0,
            mock_class_1_four=3.0
        )
    ]

    return MCMCSamples(
        model=model,
        samples=samples,
        auto_correlations=AutoCorrelations(
            times=1,
            check_size=2,
            required_length=3,
            change_threshold=4,
            previous_times=5
        ),
        total_walkers=5,
        total_steps=6,
        time=7,
    )

