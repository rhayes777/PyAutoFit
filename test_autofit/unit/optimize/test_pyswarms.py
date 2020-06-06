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
        config_path=os.path.join(directory, "files/pyswarms/config"),
        output_path=os.path.join(directory, "files/pyswarms/output"),
    )


class TestPySwarmsGlobalConfig:
    def test__loads_from_config_file_correct(self):

        pso = af.PySwarmsGlobal(
            n_particles=51,
            iters=2001,
            cognitive=0.4,
            social=0.5,
            inertia=0.6,
            initialize_method="ball",
            initialize_ball_lower_limit=0.2,
            initialize_ball_upper_limit=0.8,
            number_of_cores=2,
        )

        assert pso.n_particles == 51
        assert pso.iters == 2001
        assert pso.cognitive == 0.4
        assert pso.social == 0.5
        assert pso.inertia == 0.6
        assert pso.initialize_method == "ball"
        assert pso.initialize_ball_lower_limit == 0.2
        assert pso.initialize_ball_upper_limit == 0.8
        assert pso.number_of_cores == 2

        pso = af.PySwarmsGlobal()

        assert pso.n_particles == 50
        assert pso.iters == 2000
        assert pso.cognitive == 0.1
        assert pso.social == 0.2
        assert pso.inertia == 0.3
        assert pso.initialize_method == "prior"
        assert pso.initialize_ball_lower_limit == 0.49
        assert pso.initialize_ball_upper_limit == 0.51
        assert pso.number_of_cores == 1

    def test__samples_from_model(self):

        pyswarms = af.PySwarmsGlobal(paths=af.Paths())

        model = af.ModelMapper(mock_class=MockClassNLOx4)
        model.mock_class.two = af.LogUniformPrior(lower_limit=0.0, upper_limit=10.0)

        samples = pyswarms.samples_from_model(model=model)

        assert samples.parameters[0] == pytest.approx(
            [0.173670, 0.162607, 3095.28, 0.62104], 1.0e-4
        )
        assert samples.log_likelihoods[0] == pytest.approx(-17257775239.32677, 1.0e-4)
        assert samples.log_priors[0] == pytest.approx(1.6102016075510708, 1.0e-4)
        assert samples.weights[0] == pytest.approx(1.0, 1.0e-4)
        assert samples.total_steps == 1000
        assert samples.total_walkers == 10
        assert samples.auto_correlation_times[0] == pytest.approx(31.98507, 1.0e-4)


class TestPySwarmsGlobalOutput:
    def test__most_probable_parameters(self):

        pyswarms = af.PySwarmsGlobal(paths=af.Paths())

        model = af.ModelMapper(mock_class=MockClassNLOx4)
        model.mock_class.two = af.LogUniformPrior(lower_limit=0.0, upper_limit=10.0)

        samples = pyswarms.samples_from_model(model=model)

        assert samples.most_probable_vector == pytest.approx(
            [0.008422, -0.026413, 9.9579656, 0.494618], 1.0e-3
        )

    def test__vector_at_sigma__uses_output_files(self):

        pyswarms = af.PySwarmsGlobal(paths=af.Paths())

        model = af.ModelMapper(mock_class=MockClassNLOx4)
        model.mock_class.two = af.LogUniformPrior(lower_limit=0.0, upper_limit=10.0)

        samples = pyswarms.samples_from_model(model=model)

        params = samples.vector_at_sigma(sigma=3.0)

        assert params[0][0:2] == pytest.approx((-0.003197, 0.019923), 1e-2)

        params = samples.vector_at_sigma(sigma=1.0)

        assert params[0][0:2] == pytest.approx((0.0042278, 0.01087681), 1e-2)

    def test__autocorrelation_times(self):

        pyswarms = af.PySwarmsGlobal(paths=af.Paths())

        model = af.ModelMapper(mock_class=MockClassNLOx4)
        model.mock_class.two = af.LogUniformPrior(lower_limit=0.0, upper_limit=10.0)

        samples = pyswarms.samples_from_model(model=model)

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

    def test__pyswarms(self):
        optimizer = af.PySwarmsGlobal(af.Paths("phase_name"), sigma=2.0)

        copy = optimizer.copy_with_name_extension("one")
        self.assert_non_linear_attributes_equal(copy)
        assert isinstance(copy, af.PySwarmsGlobal)
        assert copy.sigma is optimizer.sigma
        assert copy.n_particles is optimizer.n_particles
        assert copy.iters is optimizer.iters
        assert copy.cognitive == optimizer.cognitive
        assert copy.social == optimizer.social
        assert copy.inertia == optimizer.inertia
        assert copy.ftol is optimizer.ftol
        assert copy.initialize_method is optimizer.initialize_method
        assert copy.initialize_ball_lower_limit is optimizer.initialize_ball_lower_limit
        assert copy.initialize_ball_upper_limit is optimizer.initialize_ball_upper_limit
        assert (
            copy.number_of_cores
            is optimizer.number_of_cores
        )