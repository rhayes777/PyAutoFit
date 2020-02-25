import os
import shutil
from functools import wraps

import pytest

import autofit as af
from autofit import Paths
from autofit.optimize.non_linear.output import AbstractOutput
from test_autofit.mock import MockClassNLOx4, MockClassNLOx5, MockClassNLOx6

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


class MockOutput(AbstractOutput):
    def __init__(
        self,
        model,
        paths,
        most_probable_vector=None,
        most_likely_vector=None,
        vector_at_sigma=None,
        sample_vector=None,
    ):

        super(MockOutput, self).__init__(model=model, paths=paths)

        self._most_probable_vector = most_probable_vector
        self._most_likely_vector = most_likely_vector
        self._vector_at_sigma = vector_at_sigma
        self._sample_vector = sample_vector

    @property
    def most_probable_vector(self):
        return self._most_probable_vector

    @property
    def most_likely_vector(self):
        return self._most_likely_vector

    def vector_at_sigma(self, sigma):
        return [(sigma * value[0], sigma * value[1]) for value in self._vector_at_sigma]

    def vector_from_sample_index(self, sample_index):
        return self._sample_vector[sample_index]


class TestOutput:
    def test__most_probable_instance(self):

        model = af.ModelMapper(mock_class_1=MockClassNLOx4, mock_class_2=MockClassNLOx6)

        output = MockOutput(
            model=model,
            paths=Paths(),
            most_probable_vector=[
                1.0,
                2.0,
                3.0,
                4.0,
                -5.0,
                -6.0,
                -7.0,
                -8.0,
                9.0,
                10.0,
            ],
        )

        most_probable = output.most_probable_instance

        assert most_probable.mock_class_1.one == 1.0
        assert most_probable.mock_class_1.two == 2.0
        assert most_probable.mock_class_1.three == 3.0
        assert most_probable.mock_class_1.four == 4.0

        assert most_probable.mock_class_2.one == (-5.0, -6.0)
        assert most_probable.mock_class_2.two == (-7.0, -8.0)
        assert most_probable.mock_class_2.three == 9.0
        assert most_probable.mock_class_2.four == 10.0

    def test__most_likely_instance(self):

        model = af.ModelMapper(mock_class_1=MockClassNLOx4, mock_class_2=MockClassNLOx6)

        output = MockOutput(
            model=model,
            paths=Paths(),
            most_likely_vector=[
                21.0,
                22.0,
                23.0,
                24.0,
                25.0,
                -26.0,
                -27.0,
                28.0,
                29.0,
                30.0,
            ],
        )

        most_likely = output.most_likely_instance

        assert most_likely.mock_class_1.one == 21.0
        assert most_likely.mock_class_1.two == 22.0
        assert most_likely.mock_class_1.three == 23.0
        assert most_likely.mock_class_1.four == 24.0

        assert most_likely.mock_class_2.one == (25.0, -26.0)
        assert most_likely.mock_class_2.two == (-27.0, 28.0)
        assert most_likely.mock_class_2.three == 29.0
        assert most_likely.mock_class_2.four == 30.0

    def test__vector_at_upper_and_lower_sigma(self,):

        model = af.ModelMapper(mock_class=MockClassNLOx4)
        output = MockOutput(
            model=model,
            paths=Paths(),
            most_probable_vector=[1.0, 2.0, 3.0, 4.1],
            vector_at_sigma=[(0.88, 1.12), (1.88, 2.12), (2.88, 3.12), (3.88, 4.12)],
        )

        params_upper = output.vector_at_upper_sigma(sigma=1.0)
        assert params_upper == pytest.approx([1.12, 2.12, 3.12, 4.12], 1e-2)
        params_lower = output.vector_at_lower_sigma(sigma=1.0)
        assert params_lower == pytest.approx([0.88, 1.88, 2.88, 3.88], 1e-2)

        params_upper = output.vector_at_upper_sigma(sigma=2.0)
        assert params_upper == pytest.approx(
            [2.0 * 1.12, 2.0 * 2.12, 2.0 * 3.12, 2.0 * 4.12], 1e-2
        )
        params_lower = output.vector_at_lower_sigma(sigma=2.0)
        assert params_lower == pytest.approx(
            [2.0 * 0.88, 2.0 * 1.88, 2.0 * 2.88, 2.0 * 3.88], 1e-2
        )

        instance = output.instance_at_upper_sigma(sigma=1.0)
        assert instance.mock_class.one == pytest.approx(1.12, 1e-2)
        assert instance.mock_class.two == pytest.approx(2.12, 1e-2)
        assert instance.mock_class.three == pytest.approx(3.12, 1e-2)
        assert instance.mock_class.four == pytest.approx(4.12, 1e-2)

        instance = output.instance_at_lower_sigma(sigma=1.0)
        assert instance.mock_class.one == pytest.approx(0.88, 1e-2)
        assert instance.mock_class.two == pytest.approx(1.88, 1e-2)
        assert instance.mock_class.three == pytest.approx(2.88, 1e-2)
        assert instance.mock_class.four == pytest.approx(3.88, 1e-2)

    def test__gaussian_priors(self):

        model = af.ModelMapper(mock_class=MockClassNLOx4)
        output = MockOutput(
            model=model,
            paths=Paths(),
            most_probable_vector=[1.0, 2.0, 3.0, 4.1],
            vector_at_sigma=[(0.88, 1.12), (1.88, 2.12), (2.88, 3.12), (3.88, 4.12)],
        )

        gaussian_priors = output.gaussian_priors_at_sigma(sigma=1.0)

        assert gaussian_priors[0][0] == 1.0
        assert gaussian_priors[1][0] == 2.0
        assert gaussian_priors[2][0] == 3.0
        assert gaussian_priors[3][0] == 4.1

        assert gaussian_priors[0][1] == pytest.approx(0.12, 1e-2)
        assert gaussian_priors[1][1] == pytest.approx(0.12, 1e-2)
        assert gaussian_priors[2][1] == pytest.approx(0.12, 1e-2)
        assert gaussian_priors[3][1] == pytest.approx(0.22, 1e-2)

    def test__offset_vector_from_input_vector(self):

        model = af.ModelMapper(mock_class=MockClassNLOx4)
        output = MockOutput(
            model=model, paths=Paths(), most_probable_vector=[1.0, -2.0, 3.0, 4.0]
        )

        offset_values = output.offset_vector_from_input_vector(
            input_vector=[1.0, 1.0, 2.0, 3.0]
        )

        assert offset_values == [0.0, -3.0, 1.0, 1.0]

    def test__instance_from_sample_index(self,):

        model = af.ModelMapper(mock_class=MockClassNLOx4)
        output = MockOutput(
            model=model,
            paths=Paths(),
            sample_vector=[[1.0, -2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        )

        instance = output.instance_from_sample_index(sample_index=0)

        assert instance.mock_class.one == 1.0
        assert instance.mock_class.two == -2.0
        assert instance.mock_class.three == 3.0
        assert instance.mock_class.four == 4.0

        instance = output.instance_from_sample_index(sample_index=1)

        assert instance.mock_class.one == 5.0
        assert instance.mock_class.two == 6.0
        assert instance.mock_class.three == 7.0
        assert instance.mock_class.four == 8.0

    def test__error_vector_at_sigma(self):
        model = af.ModelMapper(mock_class=MockClassNLOx4)
        output = MockOutput(
            model=model,
            paths=Paths(),
            vector_at_sigma=[(0.88, 1.12), (1.88, 2.12), (2.88, 3.12), (3.88, 4.12)],
        )

        errors = output.error_vector_at_sigma(sigma=1.0)
        assert errors == pytest.approx(
            [1.12 - 0.88, 2.12 - 1.88, 3.12 - 2.88, 4.12 - 3.88], 1e-2
        )

        errors_instance = output.error_instance_at_sigma(sigma=1.0)
        assert errors_instance.mock_class.one == pytest.approx(1.12 - 0.88, 1e-2)
        assert errors_instance.mock_class.two == pytest.approx(2.12 - 1.88, 1e-2)
        assert errors_instance.mock_class.three == pytest.approx(3.12 - 2.88, 1e-2)
        assert errors_instance.mock_class.four == pytest.approx(4.12 - 3.88, 1e-2)

        errors = output.error_vector_at_sigma(sigma=2.0)
        assert errors == pytest.approx(
            [
                2.0 * (1.12 - 0.88),
                2.0 * (2.12 - 1.88),
                2.0 * (3.12 - 2.88),
                2.0 * (4.12 - 3.88),
            ],
            1e-2,
        )

    def test__errors_at_upper_and_lower_sigma(self):

        model = af.ModelMapper(mock_class=MockClassNLOx4)
        output = MockOutput(
            model=model,
            paths=Paths(),
            most_probable_vector=[1.1, 2.0, 3.0, 4.0],
            vector_at_sigma=[(0.88, 1.12), (1.88, 2.12), (2.88, 3.12), (3.88, 4.12)],
        )

        upper_errors = output.error_vector_at_upper_sigma(sigma=1.0)
        assert upper_errors == pytest.approx([0.02, 0.12, 0.12, 0.12], 1e-2)

        errors_instance = output.error_instance_at_upper_sigma(sigma=1.0)
        assert errors_instance.mock_class.one == pytest.approx(0.02, 1e-2)
        assert errors_instance.mock_class.two == pytest.approx(0.12, 1e-2)
        assert errors_instance.mock_class.three == pytest.approx(0.12, 1e-2)
        assert errors_instance.mock_class.four == pytest.approx(0.12, 1e-2)

        upper_errors = output.error_vector_at_upper_sigma(sigma=2.0)
        assert upper_errors == pytest.approx([1.14, 2.24, 3.24, 4.24], 1e-2)

        lower_errors = output.error_vector_at_lower_sigma(sigma=1.0)
        assert lower_errors == pytest.approx([0.22, 0.12, 0.12, 0.12], 1e-2)

        errors_instance = output.error_instance_at_lower_sigma(sigma=1.0)
        assert errors_instance.mock_class.one == pytest.approx(0.22, 1e-2)
        assert errors_instance.mock_class.two == pytest.approx(0.12, 1e-2)
        assert errors_instance.mock_class.three == pytest.approx(0.12, 1e-2)
        assert errors_instance.mock_class.four == pytest.approx(0.12, 1e-2)

        lower_errors = output.error_vector_at_lower_sigma(sigma=0.5)
        assert lower_errors == pytest.approx([0.66, 1.06, 1.56, 2.06], 1e-2)
