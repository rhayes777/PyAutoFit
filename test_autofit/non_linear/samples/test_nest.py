import pytest

import autofit as af
from autofit.mock.mock import MockClassx4, MockSamples, MockNestSamples
from autofit.non_linear.samples import Sample

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

    return MockSamples(
        model=model,
        sample_list=Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=[1.0, 2.0, 3.0, 10.0, 5.0],
            log_prior_list=[0.0, 0.0, 0.0, 0.0, 0.0],
            weight_list=[1.0, 1.0, 1.0, 1.0, 1.0],
        ),
    )


def test__samples_within_parameter_range(samples):
    model = af.ModelMapper(mock_class_1=MockClassx4)

    parameters = [
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [21.0, 22.0, 23.0, 24.0],
        [0.0, 1.0, 2.0, 3.0],
    ]

    samples = MockNestSamples(
        model=model,
        sample_list=Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=[1.0, 2.0, 3.0, 10.0, 5.0],
            log_prior_list=[0.0, 0.0, 0.0, 0.0, 0.0],
            weight_list=[1.0, 1.0, 1.0, 1.0, 1.0],
        ),
        total_samples=10,
        log_evidence=0.0,
        number_live_points=5,
    )

    samples_range = samples.samples_within_parameter_range(
        parameter_index=0, parameter_range=[-1.0, 100.0]
    )

    assert len(samples_range.parameter_lists) == 5
    assert samples.parameter_lists[0] == samples_range.parameter_lists[0]

    samples_range = samples.samples_within_parameter_range(
        parameter_index=0, parameter_range=[1.0, 100.0]
    )

    assert len(samples_range.parameter_lists) == 1
    assert samples_range.parameter_lists[0] == [21.0, 22.0, 23.0, 24.0]

    samples_range = samples.samples_within_parameter_range(
        parameter_index=2, parameter_range=[1.5, 2.5]
    )

    assert len(samples_range.parameter_lists) == 4
    assert samples_range.parameter_lists[0] == [0.0, 1.0, 2.0, 3.0]
    assert samples_range.parameter_lists[1] == [0.0, 1.0, 2.0, 3.0]
    assert samples_range.parameter_lists[2] == [0.0, 1.0, 2.0, 3.0]
    assert samples_range.parameter_lists[3] == [0.0, 1.0, 2.0, 3.0]


def test__acceptance_ratio_is_correct():
    model = af.ModelMapper(mock_class_1=MockClassx4)

    samples = MockNestSamples(
        model=model,
        sample_list=Sample.from_lists(
            model=model,
            parameter_lists=5 * [[]],
            log_likelihood_list=[1.0, 2.0, 3.0, 4.0, 5.0],
            log_prior_list=5 * [0.0],
            weight_list=5 * [0.0],
        ),
        total_samples=10,
        log_evidence=0.0,
        number_live_points=5,
    )

    assert samples.acceptance_ratio == 0.5
