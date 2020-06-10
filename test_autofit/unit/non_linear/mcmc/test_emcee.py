import os
import pytest

from autoconf import conf
import autofit as af
from test_autofit.mock import MockClassNLOx4

directory = os.path.dirname(os.path.realpath(__file__))
pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        config_path=os.path.join(directory, "files/emcee/config"),
        output_path=os.path.join(directory, "files/emcee/output"),
    )


class TestEmceeConfig:
    def test__loads_from_config_file_correct(self):

        emcee = af.Emcee(
            nwalkers=51,
            nsteps=2001,
            initialize_method="ball",
            initialize_ball_lower_limit=0.2,
            initialize_ball_upper_limit=0.8,
            auto_correlation_check_for_convergence=False,
            auto_correlation_check_size=101,
            auto_correlation_required_length=51,
            auto_correlation_change_threshold=0.02,
            number_of_cores=2,
        )

        assert emcee.nwalkers == 51
        assert emcee.nsteps == 2001
        assert emcee.initialize_method == "ball"
        assert emcee.initialize_ball_lower_limit == 0.2
        assert emcee.initialize_ball_upper_limit == 0.8
        assert emcee.auto_correlation_check_for_convergence == False
        assert emcee.auto_correlation_check_size == 101
        assert emcee.auto_correlation_required_length == 51
        assert emcee.auto_correlation_change_threshold == 0.02
        assert emcee.number_of_cores == 2

        emcee = af.Emcee()

        assert emcee.nwalkers == 50
        assert emcee.nsteps == 2000
        assert emcee.initialize_method == "prior"
        assert emcee.initialize_ball_lower_limit == 0.49
        assert emcee.initialize_ball_upper_limit == 0.51
        assert emcee.auto_correlation_check_for_convergence == True
        assert emcee.auto_correlation_check_size == 100
        assert emcee.auto_correlation_required_length == 50
        assert emcee.auto_correlation_change_threshold == 0.01
        assert emcee.number_of_cores == 1

    def test__tag(self):

        emcee = af.Emcee(
            nwalkers=11
        )

        assert emcee.tag == "emcee__nwalkers_11"

    def test__samples_from_model(self):

        emcee = af.Emcee(paths=af.Paths())
        emcee.paths.backup()

        model = af.ModelMapper(mock_class=MockClassNLOx4)
        model.mock_class.two = af.LogUniformPrior(lower_limit=0.0, upper_limit=10.0)

        samples = emcee.samples_from_model(model=model)

        assert samples.parameters[0] == pytest.approx(
            [0.173670, 0.162607, 3095.28, 0.62104], 1.0e-4
        )
        assert samples.log_likelihoods[0] == pytest.approx(-17257775239.32677, 1.0e-4)
        assert samples.log_priors[0] == pytest.approx(1.6102016075510708, 1.0e-4)
        assert samples.weights[0] == pytest.approx(1.0, 1.0e-4)
        assert samples.total_steps == 1000
        assert samples.total_walkers == 10
        assert samples.auto_correlation_times[0] == pytest.approx(31.98507, 1.0e-4)


class TestEmceeOutput:
    def test__median_pdf_parameters(self):

        emcee = af.Emcee(paths=af.Paths())

        model = af.ModelMapper(mock_class=MockClassNLOx4)
        model.mock_class.two = af.LogUniformPrior(lower_limit=0.0, upper_limit=10.0)

        samples = emcee.samples_from_model(model=model)

        assert samples.median_pdf_vector == pytest.approx(
            [0.008422, -0.026413, 9.9579656, 0.494618], 1.0e-3
        )

    def test__vector_at_sigma__uses_output_files(self):

        emcee = af.Emcee(paths=af.Paths())

        model = af.ModelMapper(mock_class=MockClassNLOx4)
        model.mock_class.two = af.LogUniformPrior(lower_limit=0.0, upper_limit=10.0)

        samples = emcee.samples_from_model(model=model)

        params = samples.vector_at_sigma(sigma=3.0)

        assert params[0][0:2] == pytest.approx((-0.003197, 0.019923), 1e-2)

        params = samples.vector_at_sigma(sigma=1.0)

        assert params[0][0:2] == pytest.approx((0.0042278, 0.01087681), 1e-2)

    def test__autocorrelation_times(self):

        emcee = af.Emcee(paths=af.Paths())

        model = af.ModelMapper(mock_class=MockClassNLOx4)
        model.mock_class.two = af.LogUniformPrior(lower_limit=0.0, upper_limit=10.0)

        samples = emcee.samples_from_model(model=model)

        assert samples.previous_auto_correlation_times == pytest.approx(
            [31.1079, 36.0910, 72.44768, 65.86194], 1.0e-4
        )
        assert samples.auto_correlation_times == pytest.approx(
            [31.98507, 36.51001, 73.47629, 67.67495], 1.0e-4
        )


class TestCopyWithNameExtension:
    @staticmethod
    def assert_non_linear_attributes_equal(copy):
        assert copy.paths.name == "phase_name/one"

    def test_emcee(self):
        search = af.Emcee(af.Paths("phase_name"), sigma=2.0)

        copy = search.copy_with_name_extension("one")
        self.assert_non_linear_attributes_equal(copy)
        assert isinstance(copy, af.Emcee)
        assert copy.sigma is search.sigma
        assert copy.nwalkers is search.nwalkers
        assert copy.nsteps is search.nsteps
        assert copy.initialize_method is search.initialize_method
        assert copy.initialize_ball_lower_limit is search.initialize_ball_lower_limit
        assert copy.initialize_ball_upper_limit is search.initialize_ball_upper_limit
        assert copy.auto_correlation_check_for_convergence is search.auto_correlation_check_for_convergence
        assert copy.auto_correlation_check_size is search.auto_correlation_check_size
        assert (
            copy.auto_correlation_required_length
            is search.auto_correlation_required_length
        )
        assert (
            copy.auto_correlation_change_threshold
            is search.auto_correlation_change_threshold
        )
        assert (
            copy.number_of_cores
            is search.number_of_cores
        )