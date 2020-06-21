import os
import pytest

from autoconf import conf
import autofit as af
from test_autofit.mock import MockClassx4

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
            iterations_per_update=10,
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
        assert pso.iterations_per_update == 10
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
        assert pso.iterations_per_update == 11
        assert pso.number_of_cores == 1

    def test__tag(self):

        pso = af.PySwarmsGlobal(
            n_particles=51,
            iters=2001,
            cognitive=0.4,
            social=0.5,
            inertia=0.6,
        )

        assert pso.tag == "pyswarms__particles_51_c_0.4_s_0.5_i_0.6"

    def test__samples_from_model(self):

        pyswarms = af.PySwarmsGlobal(paths=af.Paths())

        model = af.ModelMapper(mock_class=MockClassx4)
        model.mock_class.one = af.LogUniformPrior(lower_limit=0.0, upper_limit=100.0)
        model.mock_class.two = af.LogUniformPrior(lower_limit=0.0, upper_limit=100.0)
        model.mock_class.three = af.LogUniformPrior(lower_limit=0.0, upper_limit=100.0)
        model.mock_class.four = af.LogUniformPrior(lower_limit=0.0, upper_limit=100.0)

        samples = pyswarms.samples_from_model(model=model)

        assert isinstance(samples.parameters, list)
        assert isinstance(samples.parameters[0], list)
        assert isinstance(samples.log_likelihoods, list)
        assert isinstance(samples.log_priors, list)
        assert isinstance(samples.log_posteriors, list)

        assert samples.parameters[0] == pytest.approx(
            [50.1254, 1.04626, 10.09456], 1.0e-4
        )

        assert samples.log_likelihoods[0] == pytest.approx(-5071.80777, 1.0e-4)
        assert samples.log_posteriors[0] == pytest.approx(-5070.73298, 1.0e-4)

        assert len(samples.parameters) == 500
        assert len(samples.log_likelihoods) == 500

class TestCopyWithNameExtension:
    @staticmethod
    def assert_non_linear_attributes_equal(copy):
        assert copy.paths.name == "phase_name/one"

    def test__pyswarms(self):
        search = af.PySwarmsGlobal(af.Paths("phase_name"), sigma=2.0)

        copy = search.copy_with_name_extension("one")
        self.assert_non_linear_attributes_equal(copy)
        assert isinstance(copy, af.PySwarmsGlobal)
        assert copy.sigma is search.sigma
        assert copy.n_particles is search.n_particles
        assert copy.iters is search.iters
        assert copy.cognitive == search.cognitive
        assert copy.social == search.social
        assert copy.inertia == search.inertia
        assert copy.ftol is search.ftol
        assert copy.initialize_method is search.initialize_method
        assert copy.initialize_ball_lower_limit is search.initialize_ball_lower_limit
        assert copy.initialize_ball_upper_limit is search.initialize_ball_upper_limit
        assert copy.iterations_per_update is search.iterations_per_update
        assert (
            copy.number_of_cores
            is search.number_of_cores
        )