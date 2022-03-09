import numpy as np
import pytest

from autoconf.conf import with_config

import autofit as af

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def test__from_csv_table():

    model = af.ModelMapper(mock_class_1=af.m.MockClassx4)

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

    samples_x5 = af.PDFSamples.from_table(filename=filename, model=samples_x5.model)

    assert samples_x5.parameter_lists == [
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [21.0, 22.0, 23.0, 24.0],
        [0.0, 1.0, 2.0, 3.0],
    ]
    assert samples_x5.log_likelihood_list == [1.0, 2.0, 3.0, 10.0, 5.0]
    assert samples_x5.log_prior_list == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert samples_x5.log_posterior_list == [1.0, 2.0, 3.0, 10.0, 5.0]
    assert samples_x5.weight_list == [1.0, 1.0, 1.0, 1.0, 1.0]


def test__converged__median_pdf_vector_and_instance():
    parameters = [
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [0.9, 1.9],
        [1.1, 2.1],
    ]

    log_likelihood_list = 10 * [0.1]
    weight_list = 10 * [0.1]

    model = af.ModelMapper(mock_class=af.m.MockClassx2)
    samples_x5 = af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=10 * [0.0],
            weight_list=weight_list,
        ),
    )

    assert samples_x5.pdf_converged is True

    median_pdf_vector = samples_x5.median_pdf_vector

    assert median_pdf_vector[0] == pytest.approx(1.0, 1.0e-4)
    assert median_pdf_vector[1] == pytest.approx(2.0, 1.0e-4)

    median_pdf_instance = samples_x5.median_pdf_instance

    assert median_pdf_instance.mock_class.one == pytest.approx(1.0, 1e-1)
    assert median_pdf_instance.mock_class.two == pytest.approx(2.0, 1e-1)


def test__unconverged__median_pdf_vector():
    parameters = [
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.1, 2.1],
        [0.9, 1.9],
    ]

    log_likelihood_list = 9 * [0.0] + [1.0]
    weight_list = 9 * [0.0] + [1.0]

    model = af.ModelMapper(mock_class=af.m.MockClassx2)
    samples_x5 = af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=10 * [0.0],
            weight_list=weight_list,
        ),
    )

    assert samples_x5.pdf_converged is False

    median_pdf_vector = samples_x5.median_pdf_vector

    assert median_pdf_vector[0] == pytest.approx(0.9, 1.0e-4)
    assert median_pdf_vector[1] == pytest.approx(1.9, 1.0e-4)


@with_config("general", "model", "ignore_prior_limits", value=True)
def test__converged__vector_and_instance_at_upper_and_lower_sigma():
    parameters = [
        [0.1, 0.4],
        [0.1, 0.4],
        [0.1, 0.4],
        [0.1, 0.4],
        [0.1, 0.4],
        [0.1, 0.4],
        [0.1, 0.4],
        [0.1, 0.4],
        [0.0, 0.5],
        [0.2, 0.3],
    ]

    log_likelihood_list = list(range(10))

    weight_list = 10 * [0.1]

    model = af.ModelMapper(mock_class=af.m.MockClassx2)
    samples_x5 = af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=10 * [0.0],
            weight_list=weight_list,
        ),
    )

    assert samples_x5.pdf_converged is True

    vector_at_sigma = samples_x5.vector_at_sigma(sigma=3.0)

    assert vector_at_sigma[0] == pytest.approx((0.00121, 0.19878), 1e-1)
    assert vector_at_sigma[1] == pytest.approx((0.30121, 0.49878), 1e-1)

    vector_at_sigma = samples_x5.vector_at_upper_sigma(sigma=3.0)

    assert vector_at_sigma[0] == pytest.approx(0.19757, 1e-1)
    assert vector_at_sigma[1] == pytest.approx(0.49757, 1e-1)

    vector_at_sigma = samples_x5.vector_at_lower_sigma(sigma=3.0)

    assert vector_at_sigma[0] == pytest.approx(0.00121, 1e-1)
    assert vector_at_sigma[1] == pytest.approx(0.30121, 1e-1)

    vector_at_sigma = samples_x5.vector_at_sigma(sigma=1.0)

    assert vector_at_sigma[0] == pytest.approx((0.1, 0.1), 1e-1)
    assert vector_at_sigma[1] == pytest.approx((0.4, 0.4), 1e-1)

    instance_at_sigma = samples_x5.instance_at_sigma(sigma=1.0)

    assert instance_at_sigma.mock_class.one == pytest.approx((0.1, 0.1), 1e-1)
    assert instance_at_sigma.mock_class.two == pytest.approx((0.4, 0.4), 1e-1)

    instance_at_sigma = samples_x5.instance_at_upper_sigma(sigma=3.0)

    assert instance_at_sigma.mock_class.one == pytest.approx(0.19757, 1e-1)
    assert instance_at_sigma.mock_class.two == pytest.approx(0.49757, 1e-1)

    instance_at_sigma = samples_x5.instance_at_lower_sigma(sigma=3.0)

    assert instance_at_sigma.mock_class.one == pytest.approx(0.00121, 1e-1)
    assert instance_at_sigma.mock_class.two == pytest.approx(0.30121, 1e-1)


def test__unconverged_vector_at_lower_and_upper_sigma():
    parameters = [
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.1, 2.1],
        [0.9, 1.9],
    ]

    log_likelihood_list = 9 * [0.0] + [1.0]
    weight_list = 9 * [0.0] + [1.0]

    model = af.ModelMapper(mock_class=af.m.MockClassx2)
    samples_x5 = af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=10 * [0.0],
            weight_list=weight_list,
        ),
    )

    assert samples_x5.pdf_converged is False

    vector_at_sigma = samples_x5.vector_at_sigma(sigma=1.0)

    assert vector_at_sigma[0] == pytest.approx(((0.9, 1.1)), 1e-2)
    assert vector_at_sigma[1] == pytest.approx(((1.9, 2.1)), 1e-2)

    vector_at_sigma = samples_x5.vector_at_sigma(sigma=3.0)

    assert vector_at_sigma[0] == pytest.approx(((0.9, 1.1)), 1e-2)
    assert vector_at_sigma[1] == pytest.approx(((1.9, 2.1)), 1e-2)


def test__converged__errors_vector_and_instance_at_upper_and_lower_sigma():
    parameters = [
        [0.1, 0.4],
        [0.1, 0.4],
        [0.1, 0.4],
        [0.1, 0.4],
        [0.1, 0.4],
        [0.1, 0.4],
        [0.1, 0.4],
        [0.1, 0.4],
        [0.0, 0.5],
        [0.2, 0.3],
    ]

    log_likelihood_list = list(range(10))

    weight_list = 10 * [0.1]

    model = af.ModelMapper(mock_class=af.m.MockClassx2)
    samples_x5 = af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=10 * [0.0],
            weight_list=weight_list,
        ),
    )

    assert samples_x5.pdf_converged is True

    errors = samples_x5.error_magnitude_vector_at_sigma(sigma=3.0)

    assert errors == pytest.approx([0.19514, 0.19514], 1e-1)

    errors = samples_x5.error_vector_at_upper_sigma(sigma=3.0)

    assert errors == pytest.approx([0.09757, 0.09757], 1e-1)

    errors = samples_x5.error_vector_at_lower_sigma(sigma=3.0)

    assert errors == pytest.approx([0.09757, 0.09757], 1e-1)

    errors = samples_x5.error_vector_at_sigma(sigma=3.0)
    assert errors[0] == pytest.approx((0.09757, 0.09757), 1e-1)
    assert errors[1] == pytest.approx((0.09757, 0.09757), 1e-1)

    errors = samples_x5.error_magnitude_vector_at_sigma(sigma=1.0)

    assert errors == pytest.approx([0.0, 0.0], 1e-1)

    errors_instance = samples_x5.error_instance_at_sigma(sigma=1.0)

    assert errors_instance.mock_class.one == pytest.approx(0.0, 1e-1)
    assert errors_instance.mock_class.two == pytest.approx(0.0, 1e-1)

    errors_instance = samples_x5.error_instance_at_upper_sigma(sigma=3.0)

    assert errors_instance.mock_class.one == pytest.approx(0.09757, 1e-1)
    assert errors_instance.mock_class.two == pytest.approx(0.09757, 1e-1)

    errors_instance = samples_x5.error_instance_at_lower_sigma(sigma=3.0)

    assert errors_instance.mock_class.one == pytest.approx(0.09757, 1e-1)
    assert errors_instance.mock_class.two == pytest.approx(0.09757, 1e-1)


def test__unconverged_sample_size__uses_value_unless_fewer_samples():
    model = af.ModelMapper(mock_class_1=af.m.MockClassx4)

    log_likelihood_list = 4 * [0.0] + [1.0]
    weight_list = 4 * [0.0] + [1.0]

    samples_x5 = af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=5 * [[]],
            log_likelihood_list=log_likelihood_list,
            log_prior_list=[1.0, 1.0, 1.0, 1.0, 1.0],
            weight_list=weight_list,
        ),
        unconverged_sample_size=2,
    )

    assert samples_x5.pdf_converged is False
    assert samples_x5.unconverged_sample_size == 2

    samples_x5 = af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=5 * [[]],
            log_likelihood_list=log_likelihood_list,
            log_prior_list=[1.0, 1.0, 1.0, 1.0, 1.0],
            weight_list=weight_list,
        ),
        unconverged_sample_size=6,
    )

    assert samples_x5.pdf_converged is False
    assert samples_x5.unconverged_sample_size == 5


def test__offset_vector_from_input_vector():
    model = af.ModelMapper(mock_class_1=af.m.MockClassx4)

    parameters = [
        [1.1, 2.1, 3.1, 4.1],
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.1],
    ]

    weight_list = [0.3, 0.2, 0.2, 0.2, 0.1]

    log_likelihood_list = list(map(lambda weight: 10.0 * weight, weight_list))

    samples_x5 = af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=10 * [0.0],
            weight_list=weight_list,
        ),
    )

    offset_values = samples_x5.offset_vector_from_input_vector(
        input_vector=[1.0, 1.0, 2.0, 3.0]
    )

    assert offset_values == pytest.approx([0.0, 1.0, 1.0, 1.025], 1.0e-4)


def test__vector_drawn_randomly_from_pdf():
    parameters = [
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [21.0, 22.0, 23.0, 24.0],
        [0.0, 1.0, 2.0, 3.0],
    ]

    model = af.ModelMapper(mock_class_1=af.m.MockClassx4)

    samples_x5 = af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=[1.0, 2.0, 3.0, 4.0, 5.0],
            log_prior_list=5 * [0.0],
            weight_list=[0.0, 0.0, 0.0, 1.0, 0.0],
        ),
    )

    vector = samples_x5.vector_drawn_randomly_from_pdf()

    assert vector == [21.0, 22.0, 23.0, 24.0]

    instance = samples_x5.instance_drawn_randomly_from_pdf()

    assert vector == [21.0, 22.0, 23.0, 24.0]

    assert instance.mock_class_1.one == 21.0
    assert instance.mock_class_1.two == 22.0
    assert instance.mock_class_1.three == 23.0
    assert instance.mock_class_1.four == 24.0


def test__covariance_matrix():
    log_likelihood_list = list(range(3))

    weight_list = 3 * [0.1]

    parameters = [[2.0, 2.0], [1.0, 1.0], [0.0, 0.0]]

    model = af.ModelMapper(mock_class=af.m.MockClassx2)
    samples_x5 = af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=3 * [0.0],
            weight_list=weight_list,
        ),
    )

    assert samples_x5.covariance_matrix() == pytest.approx(
        np.array([[1.0, 1.0], [1.0, 1.0]]), 1.0e-4
    )

    parameters = [[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]]

    model = af.ModelMapper(mock_class=af.m.MockClassx2)
    samples_x5 = af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=3 * [0.0],
            weight_list=weight_list,
        ),
    )

    assert samples_x5.covariance_matrix() == pytest.approx(
        np.array([[1.0, -1.0], [-1.0, 1.0]]), 1.0e-4
    )

    weight_list = [0.1, 0.2, 0.3]

    model = af.ModelMapper(mock_class=af.m.MockClassx2)
    samples_x5 = af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=10 * [0.0],
            weight_list=weight_list,
        ),
    )

    assert samples_x5.covariance_matrix() == pytest.approx(
        np.array([[0.90909, -0.90909], [-0.90909, 0.90909]]), 1.0e-4
    )
