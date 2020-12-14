import os

import pytest

import autofit as af
from autofit.mock.mock import MockClassx2, MockClassx4
from autofit.non_linear.samples import OptimizerSamples, PDFSamples, Sample

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

    return OptimizerSamples(
        model=model,
        samples=Sample.from_lists(
            model=model,
            parameters=parameters,
            log_likelihoods=[1.0, 2.0, 3.0, 10.0, 5.0],
            log_priors=[0.0, 0.0, 0.0, 0.0, 0.0],
            weights=[1.0, 1.0, 1.0, 1.0, 1.0],
        )
    )


class TestSamplesTable:
    def test_headers(self, samples):
        assert samples._headers == [
            "mock_class_1_one",
            "mock_class_1_two",
            "mock_class_1_three",
            "mock_class_1_four",
            "log_likelihood",
            "log_prior",
            "log_posterior",
            "weights",
        ]

    def test_rows(self, samples):
        rows = list(samples._rows)
        assert rows == [
            [0.0, 1.0, 2.0, 3.0, 1.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 2.0, 3.0, 2.0, 0.0, 2.0, 1.0],
            [0.0, 1.0, 2.0, 3.0, 3.0, 0.0, 3.0, 1.0],
            [21.0, 22.0, 23.0, 24.0, 10.0, 0.0, 10.0, 1.0],
            [0.0, 1.0, 2.0, 3.0, 5.0, 0.0, 5.0, 1.0],
        ]

    def test__write_table(self, samples):
        filename = "samples.csv"
        samples.write_table(filename=filename)

        assert os.path.exists(filename)
        os.remove(filename)


class TestOptimizerSamples:
    def test__max_log_likelihood_vector_and_instance(self, samples):
        assert samples.max_log_likelihood_vector == [21.0, 22.0, 23.0, 24.0]

        instance = samples.max_log_likelihood_instance

        assert instance.mock_class_1.one == 21.0
        assert instance.mock_class_1.two == 22.0
        assert instance.mock_class_1.three == 23.0
        assert instance.mock_class_1.four == 24.0

    def test__log_priors_and_max_log_posterior_vector_and_instance(self):
        model = af.ModelMapper(mock_class_1=MockClassx4)

        parameters = [
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0],
            [21.0, 22.0, 23.0, 24.0],
        ]

        samples = OptimizerSamples(
            model=model,
            samples=Sample.from_lists(
                model=model,
                parameters=parameters,
                log_likelihoods=[1.0, 2.0, 3.0, 0.0, 5.0],
                log_priors=[1.0, 2.0, 3.0, 10.0, 6.0],
                weights=[1.0, 1.0, 1.0, 1.0, 1.0],
            )
        )

        assert samples.log_posteriors == [2.0, 4.0, 6.0, 10.0, 11.0]

        assert samples.max_log_posterior_vector == [21.0, 22.0, 23.0, 24.0]

        instance = samples.max_log_posterior_instance

        assert instance.mock_class_1.one == 21.0
        assert instance.mock_class_1.two == 22.0
        assert instance.mock_class_1.three == 23.0
        assert instance.mock_class_1.four == 24.0

    def test__gaussian_priors(self):
        parameters = [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.1],
            [1.0, 2.0, 3.0, 4.1],
            [0.88, 1.88, 2.88, 3.88],
            [1.12, 2.12, 3.12, 4.32],
        ]

        model = af.ModelMapper(mock_class=MockClassx4)
        samples = OptimizerSamples(
            model=model,
            samples=Sample.from_lists(
                model=model,
                parameters=parameters,
                log_likelihoods=[10.0, 0.0, 0.0, 0.0, 0.0],
                log_priors=[0.0, 0.0, 0.0, 0.0, 0.0],
                weights=[1.0, 1.0, 1.0, 1.0, 1.0],

            ))

        gaussian_priors = samples.gaussian_priors_at_sigma(sigma=1.0)

        assert gaussian_priors[0][0] == 1.0
        assert gaussian_priors[1][0] == 2.0
        assert gaussian_priors[2][0] == 3.0
        assert gaussian_priors[3][0] == 4.0

        assert gaussian_priors[0][1] == 0.0
        assert gaussian_priors[1][1] == 0.0
        assert gaussian_priors[2][1] == 0.0
        assert gaussian_priors[3][1] == 0.0

    def test__instance_from_sample_index(self):
        model = af.ModelMapper(mock_class=MockClassx4)

        parameters = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.1, 2.1, 3.1, 4.1],
        ]

        samples = OptimizerSamples(
            model=model,
            samples=Sample.from_lists(
                model=model,
                parameters=parameters,
                log_likelihoods=[0.0, 0.0, 0.0, 0.0, 0.0],
                log_priors=[0.0, 0.0, 0.0, 0.0, 0.0],
                weights=[1.0, 1.0, 1.0, 1.0, 1.0],

            ))

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

class TestPDFSamples:
    def test__from_csv_table(self, samples):
        filename = "samples.csv"
        samples.write_table(filename=filename)

        samples = af.NestSamples.from_table(filename=filename, model=samples.model)

        assert samples.parameters == [
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0],
            [21.0, 22.0, 23.0, 24.0],
            [0.0, 1.0, 2.0, 3.0],
        ]
        assert samples.log_likelihoods == [1.0, 2.0, 3.0, 10.0, 5.0]
        assert samples.log_priors == [0.0, 0.0, 0.0, 0.0, 0.0]
        assert samples.log_posteriors == [1.0, 2.0, 3.0, 10.0, 5.0]
        assert samples.weights == [1.0, 1.0, 1.0, 1.0, 1.0]

    def test__converged__median_pdf_vector_and_instance(self):
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

        log_likelihoods = 10 * [0.1]
        weights = 10 * [0.1]

        model = af.ModelMapper(mock_class=MockClassx2)
        samples = PDFSamples(
            model=model,
            samples=Sample.from_lists(
                model=model,
                parameters=parameters,
                log_likelihoods=log_likelihoods,
                log_priors=10 * [0.0],
                weights=weights,
            ))

        assert samples.pdf_converged == True

        median_pdf_vector = samples.median_pdf_vector

        assert median_pdf_vector[0] == pytest.approx(1.0, 1.0e-4)
        assert median_pdf_vector[1] == pytest.approx(2.0, 1.0e-4)

        median_pdf_instance = samples.median_pdf_instance

        assert median_pdf_instance.mock_class.one == pytest.approx(1.0, 1e-1)
        assert median_pdf_instance.mock_class.two == pytest.approx(2.0, 1e-1)

    def test__unconverged__median_pdf_vector(self):
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

        log_likelihoods = 9 * [0.0] + [1.0]
        weights = 9 * [0.0] + [1.0]

        model = af.ModelMapper(mock_class=MockClassx2)
        samples = PDFSamples(
            model=model,
            samples=Sample.from_lists(

                model=model,
                parameters=parameters,
                log_likelihoods=log_likelihoods,
                log_priors=10 * [0.0],
                weights=weights,
            ))

        assert samples.pdf_converged is False

        median_pdf_vector = samples.median_pdf_vector

        assert median_pdf_vector[0] == pytest.approx(0.9, 1.0e-4)
        assert median_pdf_vector[1] == pytest.approx(1.9, 1.0e-4)

    def test__converged__vector_and_instance_at_upper_and_lower_sigma(self):
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

        log_likelihoods = list(range(10))

        weights = 10 * [0.1]

        model = af.ModelMapper(mock_class=MockClassx2)
        samples = PDFSamples(
            model=model,
            samples=Sample.from_lists(
                model=model,
                parameters=parameters,
                log_likelihoods=log_likelihoods,
                log_priors=10 * [0.0],
                weights=weights,
            ))

        assert samples.pdf_converged == True

        vector_at_sigma = samples.vector_at_sigma(sigma=3.0)

        assert vector_at_sigma[0] == pytest.approx((0.00242, 0.19757), 1e-1)
        assert vector_at_sigma[1] == pytest.approx((0.30243, 0.49757), 1e-1)

        vector_at_sigma = samples.vector_at_upper_sigma(sigma=3.0)

        assert vector_at_sigma[0] == pytest.approx(0.19757, 1e-1)
        assert vector_at_sigma[1] == pytest.approx(0.49757, 1e-1)

        vector_at_sigma = samples.vector_at_lower_sigma(sigma=3.0)

        assert vector_at_sigma[0] == pytest.approx(0.00242, 1e-1)
        assert vector_at_sigma[1] == pytest.approx(0.30243, 1e-1)

        vector_at_sigma = samples.vector_at_sigma(sigma=1.0)

        assert vector_at_sigma[0] == pytest.approx((0.1, 0.1), 1e-1)
        assert vector_at_sigma[1] == pytest.approx((0.4, 0.4), 1e-1)

        instance_at_sigma = samples.instance_at_sigma(sigma=1.0)

        assert instance_at_sigma.mock_class.one == pytest.approx((0.1, 0.1), 1e-1)
        assert instance_at_sigma.mock_class.two == pytest.approx((0.4, 0.4), 1e-1)

        instance_at_sigma = samples.instance_at_upper_sigma(sigma=3.0)

        assert instance_at_sigma.mock_class.one == pytest.approx(0.19757, 1e-1)
        assert instance_at_sigma.mock_class.two == pytest.approx(0.49757, 1e-1)

        instance_at_sigma = samples.instance_at_lower_sigma(sigma=3.0)

        assert instance_at_sigma.mock_class.one == pytest.approx(0.00242, 1e-1)
        assert instance_at_sigma.mock_class.two == pytest.approx(0.30243, 1e-1)

    def test__unconverged_vector_at_lower_and_upper_sigma(self):
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

        log_likelihoods = 9 * [0.0] + [1.0]
        weights = 9 * [0.0] + [1.0]

        model = af.ModelMapper(mock_class=MockClassx2)
        samples = PDFSamples(
            model=model,
            samples=Sample.from_lists(
                model=model,
                parameters=parameters,
                log_likelihoods=log_likelihoods,
                log_priors=10 * [0.0],
                weights=weights,
            ))

        assert samples.pdf_converged == False

        vector_at_sigma = samples.vector_at_sigma(sigma=1.0)

        assert vector_at_sigma[0] == pytest.approx(((0.9, 1.1)), 1e-2)
        assert vector_at_sigma[1] == pytest.approx(((1.9, 2.1)), 1e-2)

        vector_at_sigma = samples.vector_at_sigma(sigma=3.0)

        assert vector_at_sigma[0] == pytest.approx(((0.9, 1.1)), 1e-2)
        assert vector_at_sigma[1] == pytest.approx(((1.9, 2.1)), 1e-2)

    def test__converged__errors_vector_and_instance_at_upper_and_lower_sigma(self):
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

        log_likelihoods = list(range(10))

        weights = 10 * [0.1]

        model = af.ModelMapper(mock_class=MockClassx2)
        samples = PDFSamples(
            model=model,
            samples=Sample.from_lists(
                model=model,
                parameters=parameters,
                log_likelihoods=log_likelihoods,
                log_priors=10 * [0.0],
                weights=weights,
            ))

        assert samples.pdf_converged == True

        errors = samples.error_magnitude_vector_at_sigma(sigma=3.0)

        assert errors == pytest.approx([0.19514, 0.19514], 1e-1)

        errors = samples.error_vector_at_upper_sigma(sigma=3.0)

        assert errors == pytest.approx([0.09757, 0.09757], 1e-1)

        errors = samples.error_vector_at_lower_sigma(sigma=3.0)

        assert errors == pytest.approx([0.09757, 0.09757], 1e-1)

        errors = samples.error_vector_at_sigma(sigma=3.0)
        assert errors[0] == pytest.approx((0.09757, 0.09757), 1e-1)
        assert errors[1] == pytest.approx((0.09757, 0.09757), 1e-1)

        errors = samples.error_magnitude_vector_at_sigma(sigma=1.0)

        assert errors == pytest.approx([0.0, 0.0], 1e-1)

        errors_instance = samples.error_instance_at_sigma(sigma=1.0)

        assert errors_instance.mock_class.one == pytest.approx(0.0, 1e-1)
        assert errors_instance.mock_class.two == pytest.approx(0.0, 1e-1)

        errors_instance = samples.error_instance_at_upper_sigma(sigma=3.0)

        assert errors_instance.mock_class.one == pytest.approx(0.09757, 1e-1)
        assert errors_instance.mock_class.two == pytest.approx(0.09757, 1e-1)

        errors_instance = samples.error_instance_at_lower_sigma(sigma=3.0)

        assert errors_instance.mock_class.one == pytest.approx(0.09757, 1e-1)
        assert errors_instance.mock_class.two == pytest.approx(0.09757, 1e-1)

    def test__unconverged_sample_size__uses_value_unless_fewer_samples(self):
        model = af.ModelMapper(mock_class_1=MockClassx4)

        log_likelihoods = 4 * [0.0] + [1.0]
        weights = 4 * [0.0] + [1.0]

        samples = PDFSamples(
            model=model,
            samples=Sample.from_lists(
                model=model,
                parameters=5 * [[]],
                log_likelihoods=log_likelihoods,
                log_priors=[1.0, 1.0, 1.0, 1.0, 1.0],
                weights=weights,

            ),
            unconverged_sample_size=2,
        )

        assert samples.pdf_converged == False
        assert samples.unconverged_sample_size == 2

        samples = PDFSamples(
            model=model,
            samples=Sample.from_lists(
                model=model,
                parameters=5 * [[]],
                log_likelihoods=log_likelihoods,
                log_priors=[1.0, 1.0, 1.0, 1.0, 1.0],
                weights=weights,
            ),
            unconverged_sample_size=6,
        )

        assert samples.pdf_converged == False
        assert samples.unconverged_sample_size == 5

    def test__offset_vector_from_input_vector(self):
        model = af.ModelMapper(mock_class_1=MockClassx4)

        parameters = [
            [1.1, 2.1, 3.1, 4.1],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.1],
        ]

        weights = [0.3, 0.2, 0.2, 0.2, 0.1]

        log_likelihoods = list(map(lambda weight: 10.0 * weight, weights))

        samples = PDFSamples(
            model=model,
            samples=Sample.from_lists(

                model=model,
                parameters=parameters,
                log_likelihoods=log_likelihoods,
                log_priors=10 * [0.0],
                weights=weights,
            ))

        offset_values = samples.offset_vector_from_input_vector(
            input_vector=[1.0, 1.0, 2.0, 3.0]
        )

        assert offset_values == pytest.approx([0.0, 1.0, 1.0, 1.025], 1.0e-4)


class TestNestSamples:
    def test__acceptance_ratio_is_correct(self):
        model = af.ModelMapper(mock_class_1=MockClassx4)

        samples = af.NestSamples(
            model=model,
            samples=Sample.from_lists(
                model=model,
                parameters=5 * [[]],
                log_likelihoods=[1.0, 2.0, 3.0, 4.0, 5.0],
                log_priors=5 * [0.0],
                weights=5 * [0.0],
            ),
            total_samples=10,
            log_evidence=0.0,
            number_live_points=5,
        )

        assert samples.acceptance_ratio == 0.5

    def test__samples_within_parameter_range(self, samples):
        model = af.ModelMapper(mock_class_1=MockClassx4)

        parameters = [
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0],
            [21.0, 22.0, 23.0, 24.0],
            [0.0, 1.0, 2.0, 3.0],
        ]

        samples = af.NestSamples(
            model=model,
            samples=Sample.from_lists(
                model=model,
                parameters=parameters,
                log_likelihoods=[1.0, 2.0, 3.0, 10.0, 5.0],
                log_priors=[0.0, 0.0, 0.0, 0.0, 0.0],
                weights=[1.0, 1.0, 1.0, 1.0, 1.0],
            ),
            total_samples=10,
            log_evidence=0.0,
            number_live_points=5,
        )

        samples_range = samples.samples_within_parameter_range(parameter_index=0, parameter_range=[-1.0, 100.0])

        assert len(samples_range.parameters) == 5
        assert samples.parameters[0] == samples_range.parameters[0]

        samples_range = samples.samples_within_parameter_range(parameter_index=0, parameter_range=[1.0, 100.0])

        assert len(samples_range.parameters) == 1
        assert samples_range.parameters[0] == [21.0, 22.0, 23.0, 24.0]

        samples_range = samples.samples_within_parameter_range(parameter_index=2, parameter_range=[1.5, 2.5])

        assert len(samples_range.parameters) == 4
        assert samples_range.parameters[0] == [0.0, 1.0, 2.0, 3.0]
        assert samples_range.parameters[1] == [0.0, 1.0, 2.0, 3.0]
        assert samples_range.parameters[2] == [0.0, 1.0, 2.0, 3.0]
        assert samples_range.parameters[3] == [0.0, 1.0, 2.0, 3.0]