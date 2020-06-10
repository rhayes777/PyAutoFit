import os

import pytest

import autofit as af
from autofit.non_linear.samples import OptimizerSamples, PDFSamples
from test_autofit.mock import MockClassNLOx2, MockClassNLOx4

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(
    name="samples"
)
def make_samples():
    model = af.ModelMapper(mock_class_1=MockClassNLOx4)

    parameters = [[0.0, 1.0, 2.0, 3.0],
                  [0.0, 1.0, 2.0, 3.0],
                  [0.0, 1.0, 2.0, 3.0],
                  [21.0, 22.0, 23.0, 24.0],
                  [0.0, 1.0, 2.0, 3.0]]

    return OptimizerSamples(
        model=model,
        parameters=parameters,
        log_likelihoods=[1.0, 2.0, 3.0, 10.0, 5.0],
        log_priors=[0.0, 0.0, 0.0, 0.0, 0.0]
    )


class TestSamplesTable:
    def test_headers(self, samples):
        assert samples._headers == [
            "mock_class_1_one",
            "mock_class_1_two",
            "mock_class_1_three",
            "mock_class_1_four",
            "log_posterior",
            "log_likelihood",
            "log_prior"
        ]

    def test_rows(self, samples):
        rows = list(samples._rows)
        assert rows == [[0.0, 1.0, 2.0, 3.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 2.0, 3.0, 2.0, 2.0, 0.0],
                        [0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 0.0],
                        [21.0, 22.0, 23.0, 24.0, 10.0, 10.0, 0.0],
                        [0.0, 1.0, 2.0, 3.0, 5.0, 5.0, 0.0]]

    def test_write_table(self, samples):
        filename = "table.csv"
        samples.write_table(
            filename
        )

        assert os.path.exists(filename)
        os.remove(filename)


class TestOptimizerSamples:
    def test__parameter_names_and_labels(self, samples):
        assert samples.parameter_names == [
            "mock_class_1_one",
            "mock_class_1_two",
            "mock_class_1_three",
            "mock_class_1_four",
        ]

        assert samples.parameter_labels == [
            r"x4p0_{\mathrm{a}}",
            r"x4p1_{\mathrm{a}}",
            r"x4p2_{\mathrm{a}}",
            r"x4p3_{\mathrm{a}}",
        ]

    def test__max_log_likelihood_vector_and_instance(self, samples):
        assert samples.max_log_likelihood_vector == [21.0, 22.0, 23.0, 24.0]

        instance = samples.max_log_likelihood_instance

        assert instance.mock_class_1.one == 21.0
        assert instance.mock_class_1.two == 22.0
        assert instance.mock_class_1.three == 23.0
        assert instance.mock_class_1.four == 24.0

    def test__log_priors_and_max_log_posterior_vector_and_instance(self):
        model = af.ModelMapper(mock_class_1=MockClassNLOx4)

        parameters = [[0.0, 1.0, 2.0, 3.0],
                      [0.0, 1.0, 2.0, 3.0],
                      [0.0, 1.0, 2.0, 3.0],
                      [0.0, 1.0, 2.0, 3.0],
                      [21.0, 22.0, 23.0, 24.0]]

        samples = OptimizerSamples(
            model=model,
            parameters=parameters,
            log_likelihoods=[1.0, 2.0, 3.0, 0.0, 5.0],
            log_priors=[1.0, 2.0, 3.0, 10.0, 6.0]
        )

        assert samples.log_posteriors == [2.0, 4.0, 6.0, 10.0, 11.0]

        assert samples.max_log_posterior_vector == [21.0, 22.0, 23.0, 24.0]

        instance = samples.max_log_posterior_instance

        assert instance.mock_class_1.one == 21.0
        assert instance.mock_class_1.two == 22.0
        assert instance.mock_class_1.three == 23.0
        assert instance.mock_class_1.four == 24.0

    def test__gaussian_priors(self):
        parameters = [[1.0, 2.0, 3.0, 4.0],
                      [1.0, 2.0, 3.0, 4.1],
                      [1.0, 2.0, 3.0, 4.1],
                      [0.88, 1.88, 2.88, 3.88],
                      [1.12, 2.12, 3.12, 4.32]]

        model = af.ModelMapper(mock_class=MockClassNLOx4)
        samples = OptimizerSamples(
            model=model,
            parameters=parameters,
            log_likelihoods=[10.0, 0.0, 0.0, 0.0, 0.0],
            log_priors=[0.0, 0.0, 0.0, 0.0, 0.0]
        )

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
        model = af.ModelMapper(mock_class=MockClassNLOx4)

        parameters = [[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0],
                      [1.0, 2.0, 3.0, 4.0],
                      [1.0, 2.0, 3.0, 4.0],
                      [1.1, 2.1, 3.1, 4.1]]

        samples = OptimizerSamples(
            model=model,
            parameters=parameters,
            log_likelihoods=[0.0, 0.0, 0.0, 0.0, 0.0],
            log_priors=[0.0, 0.0, 0.0, 0.0, 0.0]
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


class TestPDFSamples:

    def test__converged_vector__median_pdf_vector_and_instance(self):
        parameters = [[1.0, 2.0],
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

        log_likelihoods = list(range(10))

        model = af.ModelMapper(mock_class=MockClassNLOx2)
        samples = PDFSamples(
            model=model,
            parameters=parameters,
            log_likelihoods=log_likelihoods,
            log_priors=[],
            weights=log_likelihoods
        )

        #      assert samples.pdf_converged == True

        errors = samples.vector_at_sigma(sigma=3.0)

        assert errors[0][0:2] == pytest.approx((0.88, 1.12), 1e-1)
        assert errors[1][0:2] == pytest.approx((1.88, 2.12), 1e-1)

        errors = samples.vector_at_sigma(sigma=1.0)

        assert errors[0][0:2] == pytest.approx((0.93, 1.07), 1e-1)
        assert errors[1][0:2] == pytest.approx((1.93, 2.07), 1e-1)

    def test__converged_vector__at_upper_and_lower_sigma(self):
        parameters = [[0.1, 0.4],
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

        model = af.ModelMapper(mock_class=MockClassNLOx2)
        samples = PDFSamples(
            model=model,
            parameters=parameters,
            log_likelihoods=log_likelihoods,
            log_priors=[],
            weights=weights
        )

        #    assert samples.pdf_converged == True

        median_pdf = samples.median_pdf_instance

        assert median_pdf.mock_class.one == pytest.approx(0.1, 1.0e-4)
        assert median_pdf.mock_class.two == pytest.approx(0.4, 1.0e-4)

    def test__unconverged_sample_size__uses_value_unless_fewer_samples(self):
        model = af.ModelMapper(mock_class_1=MockClassNLOx4)

        log_likelihoods = [1.0, 2.0, 3.0, 10.0, 5.0]

        samples = PDFSamples(
            model=model,
            parameters=[[]],
            log_likelihoods=log_likelihoods,
            log_priors=[1.0, 1.0, 1.0, 1.0, 1.0],
            weights=[],
            unconverged_sample_size=2,
        )

        assert samples.pdf_converged == False

        assert samples.unconverged_sample_size == 2

        samples = PDFSamples(
            model=model,
            parameters=[[]],
            log_likelihoods=log_likelihoods,
            log_priors=[1.0, 1.0, 1.0, 1.0, 1.0],
            weights=[],
            unconverged_sample_size=6,
        )

        assert samples.unconverged_sample_size == 5

    def test__unconverged_pdf__median_pdf_vector_and_instance(self):
        model = af.ModelMapper(mock_class_1=MockClassNLOx4)

        parameters = [[1.0, 2.0, 3.0, 4.0],
                      [1.0, 2.0, 3.0, 4.0],
                      [1.0, 2.0, 3.0, 4.0],
                      [1.0, 2.0, 3.0, 4.0],
                      [1.1, 2.1, 3.1, 4.1]]

        weights = [1.0, 0.0, 0.0, 0.0, 0.0]

        log_likelihoods = list(map(lambda weight: 10.0 * weight, weights))

        samples = PDFSamples(
            model=model,
            parameters=parameters,
            log_likelihoods=log_likelihoods,
            log_priors=[],
            weights=weights
        )

        assert samples.pdf_converged == False

        median_pdf = samples.median_pdf_instance

        assert median_pdf.mock_class_1.one == pytest.approx(1.02, 1.0e-4)
        assert median_pdf.mock_class_1.two == pytest.approx(2.02, 1.0e-4)
        assert median_pdf.mock_class_1.three == pytest.approx(3.02, 1.0e-4)
        assert median_pdf.mock_class_1.four == pytest.approx(4.02, 1.0e-4)

    def test__unconverged_vector_at_upper_and_lower_sigma(self):
        parameters = [[1.0, 2.0, 3.0, 4.0],
                      [1.0, 2.0, 3.0, 4.0],
                      [1.0, 2.0, 3.0, 4.0],
                      [0.88, 1.88, 2.88, 3.88],
                      [1.12, 2.12, 3.12, 4.12]]

        weights = [1.0, 0.0, 0.0, 0.0, 0.0]

        log_likelihoods = list(map(lambda weight: 10.0 * weight, weights))

        model = af.ModelMapper(mock_class=MockClassNLOx4)
        samples = PDFSamples(
            model=model,
            parameters=parameters,
            log_likelihoods=log_likelihoods,
            log_priors=[],
            weights=weights
        )

        assert samples.pdf_converged == False

        parameters_upper = samples.vector_at_upper_sigma(sigma=1.0)
        assert parameters_upper == pytest.approx([1.12, 2.12, 3.12, 4.12], 1e-2)
        parameters_lower = samples.vector_at_lower_sigma(sigma=1.0)
        assert parameters_lower == pytest.approx([0.88, 1.88, 2.88, 3.88], 1e-2)

        parameters_upper = samples.vector_at_upper_sigma(sigma=2.0)
        assert parameters_upper == pytest.approx(
            [1.12, 2.12, 3.12, 4.12], 1e-2
        )
        parameters_lower = samples.vector_at_lower_sigma(sigma=2.0)
        assert parameters_lower == pytest.approx(
            [0.88, 1.88, 2.88, 3.88], 1e-2
        )

        instance = samples.instance_at_upper_sigma(sigma=1.0)
        assert instance.mock_class.one == pytest.approx(1.12, 1e-2)
        assert instance.mock_class.two == pytest.approx(2.12, 1e-2)
        assert instance.mock_class.three == pytest.approx(3.12, 1e-2)
        assert instance.mock_class.four == pytest.approx(4.12, 1e-2)

        instance = samples.instance_at_lower_sigma(sigma=1.0)
        assert instance.mock_class.one == pytest.approx(0.88, 1e-2)
        assert instance.mock_class.two == pytest.approx(1.88, 1e-2)
        assert instance.mock_class.three == pytest.approx(2.88, 1e-2)
        assert instance.mock_class.four == pytest.approx(3.88, 1e-2)

    def test__unconverged_error_vector_and_instance_at_sigma(self):
        parameters = [[1.0, 2.0, 3.0, 4.0],
                      [1.0, 2.0, 3.0, 4.0],
                      [1.0, 2.0, 3.0, 4.0],
                      [0.88, 1.88, 2.88, 3.88],
                      [1.12, 2.12, 3.12, 4.12]]

        weights = [1.0, 0.0, 0.0, 0.0, 0.0]

        log_likelihoods = list(map(lambda weight: 10.0 * weight, weights))

        model = af.ModelMapper(mock_class=MockClassNLOx4)
        samples = PDFSamples(
            model=model,
            parameters=parameters,
            log_likelihoods=log_likelihoods,
            log_priors=[],
            weights=weights
        )

        assert samples.pdf_converged == False

        errors = samples.error_vector_at_sigma(sigma=1.0)
        assert errors == pytest.approx(
            [1.12 - 0.88, 2.12 - 1.88, 3.12 - 2.88, 4.12 - 3.88], 1e-2
        )

        errors_instance = samples.error_instance_at_sigma(sigma=1.0)
        assert errors_instance.mock_class.one == pytest.approx(1.12 - 0.88, 1e-2)
        assert errors_instance.mock_class.two == pytest.approx(2.12 - 1.88, 1e-2)
        assert errors_instance.mock_class.three == pytest.approx(3.12 - 2.88, 1e-2)
        assert errors_instance.mock_class.four == pytest.approx(4.12 - 3.88, 1e-2)

        errors = samples.error_vector_at_sigma(sigma=2.0)
        assert errors == pytest.approx(
            [
                (1.12 - 0.88),
                (2.12 - 1.88),
                (3.12 - 2.88),
                (4.12 - 3.88),
            ],
            1e-2,
        )

    def test__unconverged_error_vector_and_insstance_at_upper_and_lower_sigma(self):
        parameters = [[1.0, 2.0, 3.0, 4.0],
                      [1.0, 2.0, 3.0, 4.0],
                      [1.0, 2.0, 3.0, 4.0],
                      [0.98, 1.88, 2.88, 3.88],
                      [1.02, 2.12, 3.12, 4.22]]

        weights = [1.0, 0.0, 0.0, 0.0, 0.0]

        log_likelihoods = list(map(lambda weight: 10.0 * weight, weights))

        model = af.ModelMapper(mock_class=MockClassNLOx4)
        samples = PDFSamples(
            model=model,
            parameters=parameters,
            log_likelihoods=log_likelihoods,
            log_priors=[],
            weights=weights
        )

        assert samples.pdf_converged == False

        upper_errors = samples.error_vector_at_upper_sigma(sigma=1.0)
        assert upper_errors == pytest.approx([0.02, 0.12, 0.12, 0.2], 1e-2)

        errors_instance = samples.error_instance_at_upper_sigma(sigma=1.0)
        assert errors_instance.mock_class.one == pytest.approx(0.02, 1e-2)
        assert errors_instance.mock_class.two == pytest.approx(0.12, 1e-2)
        assert errors_instance.mock_class.three == pytest.approx(0.12, 1e-2)
        assert errors_instance.mock_class.four == pytest.approx(0.2, 1e-2)

        lower_errors = samples.error_vector_at_lower_sigma(sigma=1.0)
        assert lower_errors == pytest.approx([0.02, 0.12, 0.12, 0.14], 1e-2)

        errors_instance = samples.error_instance_at_lower_sigma(sigma=1.0)
        assert errors_instance.mock_class.one == pytest.approx(0.02, 1e-2)
        assert errors_instance.mock_class.two == pytest.approx(0.12, 1e-2)
        assert errors_instance.mock_class.three == pytest.approx(0.12, 1e-2)
        assert errors_instance.mock_class.four == pytest.approx(0.14, 1e-2)

    def test__unonverged__gaussian_priors(self):
        parameters = [[1.0, 2.0, 3.0, 4.1],
                      [1.0, 2.0, 3.0, 4.1],
                      [1.0, 2.0, 3.0, 4.1],
                      [0.88, 1.88, 2.88, 3.88],
                      [1.12, 2.12, 3.12, 4.32]]

        weights = [1.0, 0.0, 0.0, 0.0, 0.0]

        log_likelihoods = list(map(lambda weight: 10.0 * weight, weights))

        model = af.ModelMapper(mock_class=MockClassNLOx4)
        samples = PDFSamples(
            model=model,
            parameters=parameters,
            log_likelihoods=log_likelihoods,
            log_priors=[],
            weights=weights
        )

        assert samples.pdf_converged == False

        gaussian_priors = samples.gaussian_priors_at_sigma(sigma=1.0)

        assert gaussian_priors[0][0] == 1.0
        assert gaussian_priors[1][0] == 2.0
        assert gaussian_priors[2][0] == 3.0
        assert gaussian_priors[3][0] == 4.1

        assert gaussian_priors[0][1] == pytest.approx(0.12, 1e-2)
        assert gaussian_priors[1][1] == pytest.approx(0.12, 1e-2)
        assert gaussian_priors[2][1] == pytest.approx(0.12, 1e-2)
        assert gaussian_priors[3][1] == pytest.approx(0.22, 1e-2)

    def test__offset_vector_from_input_vector(self):
        model = af.ModelMapper(mock_class_1=MockClassNLOx4)

        parameters = [[1.0, 2.0, 3.0, 4.0],
                      [1.0, 2.0, 3.0, 4.0],
                      [1.0, 2.0, 3.0, 4.0],
                      [1.0, 2.0, 3.0, 4.0],
                      [1.1, 2.1, 3.1, 4.1]]

        weights = [0.2, 0.2, 0.2, 0.2, 0.2]

        log_likelihoods = list(map(lambda weight: 10.0 * weight, weights))

        samples = PDFSamples(
            model=model,
            parameters=parameters,
            log_likelihoods=log_likelihoods,
            log_priors=[],
            weights=weights
        )

        offset_values = samples.offset_vector_from_input_vector(
            input_vector=[1.0, 1.0, 2.0, 3.0]
        )

        assert offset_values == pytest.approx([0.02, 1.02, 1.02, 1.02], 1.0e-4)


class TestNestSamples:

    def test__acceptance_ratio_is_correct(self):
        model = af.ModelMapper(mock_class_1=MockClassNLOx4)

        samples = af.NestSamples(
            model=model,
            parameters=[[]],
            log_likelihoods=[1.0, 2.0, 3.0, 4.0, 5.0],
            log_priors=[],
            weights=[],
            log_evidence=0.0,
            number_live_points=5,
            total_samples=10,
        )

        assert samples.acceptance_ratio == 0.5
