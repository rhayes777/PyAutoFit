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
        most_probable_model_parameters=None,
        most_likely_model_parameters=None,
        model_parameters_at_sigma_limit=None,
        sample_model_parameters=None,
    ):

        super(MockOutput, self).__init__(model=model, paths=paths)

        self._most_probable_model_parameters = most_probable_model_parameters
        self._most_likely_model_parameters = most_likely_model_parameters
        self._model_parameters_at_sigma_limit = model_parameters_at_sigma_limit
        self._sample_model_parameters = sample_model_parameters

    @property
    def most_probable_model_parameters(self):
        return self._most_probable_model_parameters

    @property
    def most_likely_model_parameters(self):
        return self._most_likely_model_parameters

    def model_parameters_at_sigma_limit(self, sigma_limit):
        return [
            (sigma_limit * value[0], sigma_limit * value[1])
            for value in self._model_parameters_at_sigma_limit
        ]

    def sample_model_parameters_from_sample_index(self, sample_index):
        return self._sample_model_parameters[sample_index]


class TestOutput:
    def test__most_probable_model_instance(self):

        model = af.ModelMapper(mock_class_1=MockClassNLOx4, mock_class_2=MockClassNLOx6)

        output = MockOutput(
            model=model,
            paths=Paths(),
            most_probable_model_parameters=[
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

        most_probable = output.most_probable_model_instance

        assert most_probable.mock_class_1.one == 1.0
        assert most_probable.mock_class_1.two == 2.0
        assert most_probable.mock_class_1.three == 3.0
        assert most_probable.mock_class_1.four == 4.0

        assert most_probable.mock_class_2.one == (-5.0, -6.0)
        assert most_probable.mock_class_2.two == (-7.0, -8.0)
        assert most_probable.mock_class_2.three == 9.0
        assert most_probable.mock_class_2.four == 10.0

    def test__most_likely_model_instance(self):

        model = af.ModelMapper(mock_class_1=MockClassNLOx4, mock_class_2=MockClassNLOx6)

        output = MockOutput(
            model=model,
            paths=Paths(),
            most_likely_model_parameters=[
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

        most_likely = output.most_likely_model_instance

        assert most_likely.mock_class_1.one == 21.0
        assert most_likely.mock_class_1.two == 22.0
        assert most_likely.mock_class_1.three == 23.0
        assert most_likely.mock_class_1.four == 24.0

        assert most_likely.mock_class_2.one == (25.0, -26.0)
        assert most_likely.mock_class_2.two == (-27.0, 28.0)
        assert most_likely.mock_class_2.three == 29.0
        assert most_likely.mock_class_2.four == 30.0

    def test__model_parameters_at_upper_and_lower_sigma_limit(self,):

        model = af.ModelMapper(mock_class=MockClassNLOx4)
        output = MockOutput(
            model=model,
            paths=Paths(),
            most_probable_model_parameters=[1.0, 2.0, 3.0, 4.1],
            model_parameters_at_sigma_limit=[
                (0.88, 1.12),
                (1.88, 2.12),
                (2.88, 3.12),
                (3.88, 4.12),
            ],
        )

        params_upper = output.model_parameters_at_upper_sigma_limit(sigma_limit=1.0)
        assert params_upper == pytest.approx([1.12, 2.12, 3.12, 4.12], 1e-2)
        params_lower = output.model_parameters_at_lower_sigma_limit(sigma_limit=1.0)
        assert params_lower == pytest.approx([0.88, 1.88, 2.88, 3.88], 1e-2)

        params_upper = output.model_parameters_at_upper_sigma_limit(sigma_limit=2.0)
        assert params_upper == pytest.approx(
            [2.0 * 1.12, 2.0 * 2.12, 2.0 * 3.12, 2.0 * 4.12], 1e-2
        )
        params_lower = output.model_parameters_at_lower_sigma_limit(sigma_limit=2.0)
        assert params_lower == pytest.approx(
            [2.0 * 0.88, 2.0 * 1.88, 2.0 * 2.88, 2.0 * 3.88], 1e-2
        )

    def test__gaussian_priors(self):

        model = af.ModelMapper(mock_class=MockClassNLOx4)
        output = MockOutput(
            model=model,
            paths=Paths(),
            most_probable_model_parameters=[1.0, 2.0, 3.0, 4.1],
            model_parameters_at_sigma_limit=[
                (0.88, 1.12),
                (1.88, 2.12),
                (2.88, 3.12),
                (3.88, 4.12),
            ],
        )

        gaussian_priors = output.gaussian_priors_at_sigma_limit(sigma_limit=1.0)

        assert gaussian_priors[0][0] == 1.0
        assert gaussian_priors[1][0] == 2.0
        assert gaussian_priors[2][0] == 3.0
        assert gaussian_priors[3][0] == 4.1

        assert gaussian_priors[0][1] == pytest.approx(0.12, 1e-2)
        assert gaussian_priors[1][1] == pytest.approx(0.12, 1e-2)
        assert gaussian_priors[2][1] == pytest.approx(0.12, 1e-2)
        assert gaussian_priors[3][1] == pytest.approx(0.22, 1e-2)

    def test__values_offset_from_input_model_parameters(self):

        model = af.ModelMapper(mock_class=MockClassNLOx4)
        output = MockOutput(
            model=model,
            paths=Paths(),
            most_probable_model_parameters=[1.0, -2.0, 3.0, 4.0],
        )

        offset_values = output.values_offset_from_input_model_parameters(
            input_model_parameters=[1.0, 1.0, 2.0, 3.0]
        )

        assert offset_values == [0.0, -3.0, 1.0, 1.0]

    def test__sample_model_instance_from_sample_index(self,):

        model = af.ModelMapper(mock_class=MockClassNLOx4)
        output = MockOutput(
            model=model,
            paths=Paths(),
            sample_model_parameters=[[1.0, -2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        )

        instance = output.sample_model_instance_from_sample_index(sample_index=0)

        assert instance.mock_class.one == 1.0
        assert instance.mock_class.two == -2.0
        assert instance.mock_class.three == 3.0
        assert instance.mock_class.four == 4.0

        instance = output.sample_model_instance_from_sample_index(sample_index=1)

        assert instance.mock_class.one == 5.0
        assert instance.mock_class.two == 6.0
        assert instance.mock_class.three == 7.0
        assert instance.mock_class.four == 8.0

    def test__model_errors_at_sigma_limit(self,):
        model = af.ModelMapper(mock_class=MockClassNLOx4)
        output = MockOutput(
            model=model,
            paths=Paths(),
            model_parameters_at_sigma_limit=[
                (0.88, 1.12),
                (1.88, 2.12),
                (2.88, 3.12),
                (3.88, 4.12),
            ],
        )

        model_errors = output.model_errors_at_sigma_limit(sigma_limit=1.0)
        assert model_errors == pytest.approx(
            [1.12 - 0.88, 2.12 - 1.88, 3.12 - 2.88, 4.12 - 3.88], 1e-2
        )

        model_errors_instance = output.model_errors_instance_at_sigma_limit(
            sigma_limit=1.0
        )
        assert model_errors_instance.mock_class.one == pytest.approx(1.12 - 0.88, 1e-2)
        assert model_errors_instance.mock_class.two == pytest.approx(2.12 - 1.88, 1e-2)
        assert model_errors_instance.mock_class.three == pytest.approx(
            3.12 - 2.88, 1e-2
        )
        assert model_errors_instance.mock_class.four == pytest.approx(4.12 - 3.88, 1e-2)

        model_errors = output.model_errors_at_sigma_limit(sigma_limit=2.0)
        assert model_errors == pytest.approx(
            [
                2.0 * (1.12 - 0.88),
                2.0 * (2.12 - 1.88),
                2.0 * (3.12 - 2.88),
                2.0 * (4.12 - 3.88),
            ],
            1e-2,
        )

    def test__model_errors_at_upper_and_lower_sigma_limit(self,):
        model = af.ModelMapper(mock_class=MockClassNLOx4)
        output = MockOutput(
            model=model,
            paths=Paths(),
            most_probable_model_parameters=[1.1, 2.0, 3.0, 4.0],
            model_parameters_at_sigma_limit=[
                (0.88, 1.12),
                (1.88, 2.12),
                (2.88, 3.12),
                (3.88, 4.12),
            ],
        )

        model_upper_errors = output.model_errors_at_upper_sigma_limit(sigma_limit=1.0)
        assert model_upper_errors == pytest.approx([0.02, 0.12, 0.12, 0.12], 1e-2)

        model_errors_instance = output.model_errors_instance_at_upper_sigma_limit(
            sigma_limit=1.0
        )
        assert model_errors_instance.mock_class.one == pytest.approx(0.02, 1e-2)
        assert model_errors_instance.mock_class.two == pytest.approx(0.12, 1e-2)
        assert model_errors_instance.mock_class.three == pytest.approx(0.12, 1e-2)
        assert model_errors_instance.mock_class.four == pytest.approx(0.12, 1e-2)

        model_upper_errors = output.model_errors_at_upper_sigma_limit(sigma_limit=2.0)
        assert model_upper_errors == pytest.approx([1.14, 2.24, 3.24, 4.24], 1e-2)

        model_lower_errors = output.model_errors_at_lower_sigma_limit(sigma_limit=1.0)
        assert model_lower_errors == pytest.approx([0.22, 0.12, 0.12, 0.12], 1e-2)

        model_errors_instance = output.model_errors_instance_at_lower_sigma_limit(
            sigma_limit=1.0
        )
        assert model_errors_instance.mock_class.one == pytest.approx(0.22, 1e-2)
        assert model_errors_instance.mock_class.two == pytest.approx(0.12, 1e-2)
        assert model_errors_instance.mock_class.three == pytest.approx(0.12, 1e-2)
        assert model_errors_instance.mock_class.four == pytest.approx(0.12, 1e-2)

        model_lower_errors = output.model_errors_at_lower_sigma_limit(sigma_limit=0.5)
        assert model_lower_errors == pytest.approx([0.66, 1.06, 1.56, 2.06], 1e-2)
