import os
import pytest

from autoconf import conf
import autofit as af
from autofit import Paths
from autofit.optimize.non_linear.emcee import EmceeOutput
from test_autofit.mock import MockClassNLOx4

directory = os.path.dirname(os.path.realpath(__file__))
pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        config_path=os.path.join(directory, "files/emcee/config"),
        output_path=os.path.join(directory, "files/emcee/output"),
    )


@pytest.fixture(name="emcee_output")
def test_emcee_output():
    emcee_output_path = "{}/files/emcee/".format(
        os.path.dirname(os.path.realpath(__file__))
    )

    af.conf.instance.output_path = emcee_output_path

    mapper = af.ModelMapper(mock_class_1=MockClassNLOx4)

    return EmceeOutput(
        mapper,
        Paths(),
        auto_correlation_check_size=10,
        auto_correlation_required_length=10,
        auto_correlation_change_threshold=0.01,
    )


class TestEmceeConfig:
    def test__loads_from_config_file_correct(self):

        emcee = af.Emcee()

        assert emcee.nwalkers == 50
        assert emcee.nsteps == 2000
        assert emcee.check_auto_correlation == True
        assert emcee.auto_correlation_check_size == 100
        assert emcee.auto_correlation_required_length == 50
        assert emcee.auto_correlation_change_threshold == 0.01


class TestEmceeOutput:

    def test__max_log_likelihood(self, emcee_output):

        emcee_output.model.mock_class_1.one = af.LogUniformPrior(lower_limit=0.0, upper_limit=10.0)

        assert emcee_output.max_log_likelihood_index == 9977
        assert emcee_output.max_log_likelihood == pytest.approx(581.24209, 1.0e-4)
        assert emcee_output.max_log_likelihood_vector == pytest.approx(
            [0.003825, -0.00360509, 9.957799, 0.4940334], 1.0e-3
        )

    def test__max_log_posterior(self, emcee_output):

        assert emcee_output.max_log_posterior_index == 9977
        assert emcee_output.max_log_posterior == pytest.approx(583.26625, 1.0e-4)
        assert emcee_output.max_log_posterior_vector == pytest.approx(
            [0.003825, -0.00360509, 9.957799, 0.4940334], 1.0e-3
        )

    def test__most_probable_parameters(sel, emcee_output):

        assert emcee_output.most_probable_vector == pytest.approx(
            [0.008422, -0.026413, 9.9579656, 0.494618], 1.0e-3
        )

    def test__vector_at_sigma__uses_output_files(self, emcee_output):

        params = emcee_output.vector_at_sigma(sigma=3.0)

        assert params[0][0:2] == pytest.approx((-0.003197, 0.019923), 1e-2)

        params = emcee_output.vector_at_sigma(sigma=1.0)

        assert params[0][0:2] == pytest.approx((0.0042278, 0.01087681), 1e-2)

    def test__samples__total_steps_samples__model_parameters_weight_and_log_posterior_from_sample_index(
        self, emcee_output
    ):

        emcee_output.model.mock_class_1.one = af.LogUniformPrior(lower_limit=0.0, upper_limit=10.0)

        model = emcee_output.vector_from_sample_index(sample_index=0)
        weight = emcee_output.weight_from_sample_index(sample_index=0)
        log_prior = emcee_output.log_prior_from_sample_index(sample_index=0)
        log_posterior = emcee_output.log_posterior_from_sample_index(sample_index=0)
        log_likelihood = emcee_output.log_likelihood_from_sample_index(sample_index=0)

        assert emcee_output.total_walkers == 10
        assert emcee_output.total_steps == 1000
        assert emcee_output.total_samples == 10000
        assert model == pytest.approx(
            [0.0090338, -0.05790179, 10.192579, 0.480606], 1.0e-2
        )
        assert weight == 1.0
        assert log_likelihood == pytest.approx(-17257775239 + 2.0807033, 1.0e-4)
        assert log_prior == pytest.approx(2.0807033, 1.0e-4)
        assert log_posterior == pytest.approx(-17257775239, 1.0e-4)

        #
        # assert emcee_output.total_samples == 10
        # assert model == [1.0, 2.0, 3.0, 4.0]
        # assert weight == 0.1
        # assert likelihood == -0.5 * 9999999.9

    def test__autocorrelation_times(self, emcee_output):

        assert (
            emcee_output.previous_auto_correlation_times_of_parameters
            == pytest.approx([31.92692, 36.54546, 73.33737, 67.52170], 1.0e-4)
        )
        assert emcee_output.auto_correlation_times_of_parameters == pytest.approx(
            [31.98507, 36.51001, 73.47629, 67.67495], 1.0e-4
        )

        assert emcee_output.converged == True


class TestCopyWithNameExtension:
    @staticmethod
    def assert_non_linear_attributes_equal(copy):
        assert copy.paths.phase_name == "phase_name/one"

    def test_emcee(self):
        optimizer = af.Emcee(Paths("phase_name"), sigma=2.0)

        copy = optimizer.copy_with_name_extension("one")
        self.assert_non_linear_attributes_equal(copy)
        assert isinstance(copy, af.Emcee)
        assert copy.sigma is optimizer.sigma
        assert copy.nwalkers is optimizer.nwalkers
        assert copy.nsteps is optimizer.nsteps
        assert copy.check_auto_correlation is optimizer.check_auto_correlation
        assert copy.auto_correlation_check_size is optimizer.auto_correlation_check_size
        assert (
            copy.auto_correlation_required_length
            is optimizer.auto_correlation_required_length
        )
        assert (
            copy.auto_correlation_change_threshold
            is optimizer.auto_correlation_change_threshold
        )
