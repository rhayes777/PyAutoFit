import os
import pytest

import autofit as af
from autofit.mock.mock import MockClassx4, MockSamples
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


def test__table__headers(samples):
    assert samples._headers == [
        "mock_class_1_one",
        "mock_class_1_two",
        "mock_class_1_three",
        "mock_class_1_four",
        "log_likelihood",
        "log_prior",
        "log_posterior",
        "weight",
    ]


def test__table__rows(samples):
    rows = list(samples._rows)
    assert rows == [
        [0.0, 1.0, 2.0, 3.0, 1.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 2.0, 3.0, 2.0, 0.0, 2.0, 1.0],
        [0.0, 1.0, 2.0, 3.0, 3.0, 0.0, 3.0, 1.0],
        [21.0, 22.0, 23.0, 24.0, 10.0, 0.0, 10.0, 1.0],
        [0.0, 1.0, 2.0, 3.0, 5.0, 0.0, 5.0, 1.0],
    ]


def test__table__write_table(samples):
    filename = "samples.csv"
    samples.write_table(filename=filename)

    assert os.path.exists(filename)
    os.remove(filename)


def test__max_log_likelihood_vector_and_instance(samples):
    assert samples.max_log_likelihood_vector == [21.0, 22.0, 23.0, 24.0]

    instance = samples.max_log_likelihood_instance

    assert instance.mock_class_1.one == 21.0
    assert instance.mock_class_1.two == 22.0
    assert instance.mock_class_1.three == 23.0
    assert instance.mock_class_1.four == 24.0


def test__log_prior_list_and_max_log_posterior_vector_and_instance():
    model = af.ModelMapper(mock_class_1=MockClassx4)

    parameters = [
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [21.0, 22.0, 23.0, 24.0],
    ]

    samples = MockSamples(
        model=model,
        sample_list=Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=[1.0, 2.0, 3.0, 0.0, 5.0],
            log_prior_list=[1.0, 2.0, 3.0, 10.0, 6.0],
            weight_list=[1.0, 1.0, 1.0, 1.0, 1.0],
        ),
    )

    assert samples.log_posterior_list == [2.0, 4.0, 6.0, 10.0, 11.0]

    assert samples.max_log_posterior_vector == [21.0, 22.0, 23.0, 24.0]

    instance = samples.max_log_posterior_instance

    assert instance.mock_class_1.one == 21.0
    assert instance.mock_class_1.two == 22.0
    assert instance.mock_class_1.three == 23.0
    assert instance.mock_class_1.four == 24.0


def test__gaussian_priors():
    parameters = [
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.1],
        [1.0, 2.0, 3.0, 4.1],
        [0.88, 1.88, 2.88, 3.88],
        [1.12, 2.12, 3.12, 4.32],
    ]

    model = af.ModelMapper(mock_class=MockClassx4)
    samples = MockSamples(
        model=model,
        sample_list=Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=[10.0, 0.0, 0.0, 0.0, 0.0],
            log_prior_list=[0.0, 0.0, 0.0, 0.0, 0.0],
            weight_list=[1.0, 1.0, 1.0, 1.0, 1.0],
        ),
    )

    gaussian_priors = samples.gaussian_priors_at_sigma(sigma=1.0)

    assert gaussian_priors[0][0] == 1.0
    assert gaussian_priors[1][0] == 2.0
    assert gaussian_priors[2][0] == 3.0
    assert gaussian_priors[3][0] == 4.0

    assert gaussian_priors[0][1] == pytest.approx(0.12, 1.0e-4)
    assert gaussian_priors[1][1] == pytest.approx(0.12, 1.0e-4)
    assert gaussian_priors[2][1] == pytest.approx(0.12, 1.0e-4)
    assert gaussian_priors[3][1] == pytest.approx(0.32, 1.0e-4)


def test__instance_from_sample_index():
    model = af.ModelMapper(mock_class=MockClassx4)

    parameters = [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0],
        [1.1, 2.1, 3.1, 4.1],
    ]

    samples = MockSamples(
        model=model,
        sample_list=Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=[0.0, 0.0, 0.0, 0.0, 0.0],
            log_prior_list=[0.0, 0.0, 0.0, 0.0, 0.0],
            weight_list=[1.0, 1.0, 1.0, 1.0, 1.0],
        ),
    )

    instance = samples.instance_from_sample_index(sample_index=0)

    assert instance.mock_class.one == 1.0
    assert instance.mock_class.two == 2.0
    assert instance.mock_class.three == 3.0
    assert instance.mock_class.four == 4.0

    instance = samples.instance_from_sample_index(sample_index=1)

    assert instance.mock_class.one == 5.0
    assert instance.mock_class.two == 6.0
    assert instance.mock_class.three == 7.0
    assert instance.mock_class.four == 8.0
