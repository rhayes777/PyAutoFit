import os
import pytest

import autofit as af

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def test__table__headers(samples_x5):
    assert samples_x5._headers == [
        "mock_class_1_one",
        "mock_class_1_two",
        "mock_class_1_three",
        "mock_class_1_four",
        "log_likelihood",
        "log_prior",
        "log_posterior",
        "weight",
    ]


def test__table__rows(samples_x5):
    rows = list(samples_x5._rows)
    assert rows == [
        [0.0, 1.0, 2.0, 3.0, 1.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 2.0, 3.0, 2.0, 0.0, 2.0, 1.0],
        [0.0, 1.0, 2.0, 3.0, 3.0, 0.0, 3.0, 1.0],
        [21.0, 22.0, 23.0, 24.0, 10.0, 0.0, 10.0, 1.0],
        [0.0, 1.0, 2.0, 3.0, 5.0, 0.0, 5.0, 1.0],
    ]


def test__table__write_table():

    model = af.Collection(mock_class_1=af.m.MockClassx4)

    parameters = [
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [21.0, 22.0, 23.0, 24.0],
        [0.0, 1.0, 2.0, 3.0],
    ]

    samples_x5 = af.Samples(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=[1.0, 2.0, 3.0, 10.0, 5.0],
            log_prior_list=[0.0, 0.0, 0.0, 0.0, 0.0],
            weight_list=[1.0, 1.0, 1.0, 1.0, 1.0],
        ),
    )

    filename = "samples.csv"
    samples_x5.write_table(filename=filename)

    assert os.path.exists(filename)
    os.remove(filename)


def test__max_log_likelihood_vector_and_instance(samples_x5):
    assert samples_x5.max_log_likelihood_vector == [21.0, 22.0, 23.0, 24.0]

    instance = samples_x5.max_log_likelihood_instance

    assert instance.mock_class_1.one == 21.0
    assert instance.mock_class_1.two == 22.0
    assert instance.mock_class_1.three == 23.0
    assert instance.mock_class_1.four == 24.0


def test__log_prior_list_and_max_log_posterior_vector_and_instance():
    model = af.Collection(mock_class_1=af.m.MockClassx4)

    parameters = [
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [21.0, 22.0, 23.0, 24.0],
    ]

    samples_x5 = af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=[1.0, 2.0, 3.0, 0.0, 5.0],
            log_prior_list=[1.0, 2.0, 3.0, 10.0, 6.0],
            weight_list=[1.0, 1.0, 1.0, 1.0, 1.0],
        ),
    )

    assert samples_x5.log_posterior_list == [2.0, 4.0, 6.0, 10.0, 11.0]

    assert samples_x5.max_log_posterior_vector == [21.0, 22.0, 23.0, 24.0]

    instance = samples_x5.max_log_posterior_instance

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

    model = af.Collection(mock_class=af.m.MockClassx4)
    samples_x5 = af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=[10.0, 0.0, 0.0, 0.0, 0.0],
            log_prior_list=[0.0, 0.0, 0.0, 0.0, 0.0],
            weight_list=[1.0, 1.0, 1.0, 1.0, 1.0],
        ),
    )

    gaussian_priors = samples_x5.gaussian_priors_at_sigma(sigma=1.0)

    assert gaussian_priors[0][0] == 1.0
    assert gaussian_priors[1][0] == 2.0
    assert gaussian_priors[2][0] == 3.0
    assert gaussian_priors[3][0] == 4.0

    assert gaussian_priors[0][1] == pytest.approx(0.12, 1.0e-4)
    assert gaussian_priors[1][1] == pytest.approx(0.12, 1.0e-4)
    assert gaussian_priors[2][1] == pytest.approx(0.12, 1.0e-4)
    assert gaussian_priors[3][1] == pytest.approx(0.32, 1.0e-4)


def test__instance_from_sample_index():
    model = af.Collection(mock_class=af.m.MockClassx4)

    parameters = [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0],
        [1.1, 2.1, 3.1, 4.1],
    ]

    samples_x5 = af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=[0.0, 0.0, 0.0, 0.0, 0.0],
            log_prior_list=[0.0, 0.0, 0.0, 0.0, 0.0],
            weight_list=[1.0, 1.0, 1.0, 1.0, 1.0],
        ),
    )

    instance = samples_x5.instance_from_sample_index(sample_index=0)

    assert instance.mock_class.one == 1.0
    assert instance.mock_class.two == 2.0
    assert instance.mock_class.three == 3.0
    assert instance.mock_class.four == 4.0

    instance = samples_x5.instance_from_sample_index(sample_index=1)

    assert instance.mock_class.one == 5.0
    assert instance.mock_class.two == 6.0
    assert instance.mock_class.three == 7.0
    assert instance.mock_class.four == 8.0


def test__addition_of_samples(samples_x5):

    samples = samples_x5 + samples_x5

    assert len(samples.sample_list) == 10
    assert samples.sample_list[0].log_likelihood == 1.0
    assert samples.sample_list[4].log_likelihood == 5.0
    assert samples.sample_list[5].log_likelihood == 1.0
    assert samples.sample_list[9].log_likelihood == 5.0

def test__sum_of_samples(samples_x5):

    samples = sum([samples_x5, samples_x5, samples_x5])

    assert len(samples.sample_list) == 15
    assert samples.sample_list[0].log_likelihood == 1.0
    assert samples.sample_list[4].log_likelihood == 5.0
    assert samples.sample_list[5].log_likelihood == 1.0
    assert samples.sample_list[9].log_likelihood == 5.0
    assert samples.sample_list[10].log_likelihood == 1.0
    assert samples.sample_list[14].log_likelihood == 5.0

def test__addition_of_samples__raises_error_if_model_mismatch(samples_x5):

    model = af.Collection(mock_class_1=af.m.MockClassx2)

    parameters = [
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [21.0, 22.0],
        [0.0, 1.0],
    ]

    samples_different_model = af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=[1.0, 2.0],
            log_prior_list=[0.0, 0.0],
            weight_list=[1.0, 1.0],
        ),
    )

    with pytest.raises(af.exc.SamplesException):
        samples_x5 + samples_different_model