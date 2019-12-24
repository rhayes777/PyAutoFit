import os
import shutil
from functools import wraps

import pytest

import autofit as af
from autofit import Paths
from autofit.optimize.non_linear.emcee import EmceeOutput
from test_autofit.mock import MockClassNLOx4, MockClassNLOx6

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(scope="session", autouse=True)
def do_something():
    af.conf.instance = af.conf.Config(
        "{}/../test_files/configs/non_linear".format(
            os.path.dirname(os.path.realpath(__file__))
        )
    )


@pytest.fixture(name="emcee_output")
def test_emcee_output():
    emcee_output_path = "{}/../test_files/non_linear/emcee/".format(
        os.path.dirname(os.path.realpath(__file__))
    )

    af.conf.instance.output_path = emcee_output_path

    mapper = af.ModelMapper(mock_class_1=MockClassNLOx4)

    return EmceeOutput(mapper, Paths())


class TestEmceeOutput:
    def test__maximum_log_likelihood(self, emcee_output):

        assert emcee_output.maximum_log_likelihood == pytest.approx(583.26625, 1.0e-4)

    def test__most_probable_parameters(sel, emcee_output):

        assert emcee_output.most_probable_model_parameters == pytest.approx(
            [0.0073896, -0.0173088, 9.934668, 0.4958672], 1.0e-3
        )

    def test__most_likely_parameters(self, emcee_output):

        assert emcee_output.most_likely_model_parameters == pytest.approx(
            [0.003825, -0.00360509, 9.957799, 0.4940334], 1.0e-3
        )

    def test__model_parameters_at_sigma_limit__uses_output_files(self, emcee_output):

        params = emcee_output.model_parameters_at_sigma_limit(sigma_limit=3.0)

        print(params)

        assert params[0][0:2] == pytest.approx((-0.003906, 0.01797), 1e-2)
        #   assert params[1][0:2] == pytest.approx((1.88, 2.12), 1e-2)
        #   assert params[2][0:2] == pytest.approx((2.88, 3.12), 1e-2)
        #   assert params[3][0:2] == pytest.approx((3.88, 4.12), 1e-2)

        params = emcee_output.model_parameters_at_sigma_limit(sigma_limit=1.0)

        assert params[0][0:2] == pytest.approx((-0.00036844, 0.0140165), 1e-2)

    #    assert params[1][0:2] == pytest.approx((1.93, 2.07), 1e-2)
    #    assert params[2][0:2] == pytest.approx((2.93, 3.07), 1e-2)
    #    assert params[3][0:2] == pytest.approx((3.93, 4.07), 1e-2)

    def test__samples__total_samples__model_parameters_weight_and_likelihood_from_sample_index(
        self, emcee_output
    ):

        model = emcee_output.sample_model_parameters_from_sample_index(sample_index=0)
        weight = emcee_output.sample_weight_from_sample_index(sample_index=0)
        likelihood = emcee_output.sample_likelihood_from_sample_index(sample_index=0)

        assert emcee_output.total_samples == 10000
        assert model == pytest.approx(
            [0.004281, -0.0040616, 9.944818, 0.493324], 1.0e-2
        )
        assert weight == 1.0
        assert likelihood == pytest.approx(-17257775239, 1.0e-4)

        # model = emcee_output.sample_model_parameters_from_sample_index(sample_index=5)
        # weight = emcee_output.sample_weight_from_sample_index(sample_index=5)
        # likelihood = emcee_output.sample_likelihood_from_sample_index(sample_index=5)
        #
        # assert emcee_output.total_samples == 10
        # assert model == [1.0, 2.0, 3.0, 4.0]
        # assert weight == 0.1
        # assert likelihood == -0.5 * 9999999.9
