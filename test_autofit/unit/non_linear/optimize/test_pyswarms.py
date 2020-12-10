from os import path
import pytest

from autoconf import conf
import autofit as af
from autofit.mock import mock

directory = path.dirname(path.realpath(__file__))
pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance.push(
        new_path=path.join(directory, "files", "pyswarms", "config"),
        output_path=path.join(directory, "files", "pyswarms", "output"),
    )


class TestPySwarmsGlobalConfig:
    def test__loads_from_config_file_correct(self):

        pso = af.PySwarmsGlobal(
            prior_passer=af.PriorPasser(sigma=2.0, use_errors=False, use_widths=False),
            n_particles=51,
            iters=2001,
            cognitive=0.4,
            social=0.5,
            inertia=0.6,
            initializer=af.InitializerBall(lower_limit=0.2, upper_limit=0.8),
            iterations_per_update=10,
            number_of_cores=2,
        )

        assert pso.prior_passer.sigma == 2.0
        assert pso.prior_passer.use_errors == False
        assert pso.prior_passer.use_widths == False
        assert pso.n_particles == 51
        assert pso.iters == 2001
        assert pso.cognitive == 0.4
        assert pso.social == 0.5
        assert pso.inertia == 0.6
        assert isinstance(pso.initializer, af.InitializerBall)
        assert pso.initializer.lower_limit == 0.2
        assert pso.initializer.upper_limit == 0.8
        assert pso.iterations_per_update == 10
        assert pso.number_of_cores == 2

        pso = af.PySwarmsGlobal()

        assert pso.prior_passer.sigma == 3.0
        assert pso.prior_passer.use_errors == True
        assert pso.prior_passer.use_widths == True
        assert pso.n_particles == 50
        assert pso.iters == 2000
        assert pso.cognitive == 0.1
        assert pso.social == 0.2
        assert pso.inertia == 0.3
        assert isinstance(pso.initializer, af.InitializerPrior)
        assert pso.iterations_per_update == 11
        assert pso.number_of_cores == 1

        pso = af.PySwarmsLocal(
            prior_passer=af.PriorPasser(sigma=2.0, use_errors=False, use_widths=False),
            n_particles=51,
            iters=2001,
            cognitive=0.4,
            social=0.5,
            inertia=0.6,
            number_of_k_neighbors=4,
            minkowski_p_norm=1,
            initializer=af.InitializerBall(lower_limit=0.2, upper_limit=0.8),
            iterations_per_update=10,
            number_of_cores=2,
        )

        assert pso.prior_passer.sigma == 2.0
        assert pso.prior_passer.use_errors == False
        assert pso.prior_passer.use_widths == False
        assert pso.n_particles == 51
        assert pso.iters == 2001
        assert pso.cognitive == 0.4
        assert pso.social == 0.5
        assert pso.inertia == 0.6
        assert pso.number_of_k_neighbors == 4
        assert pso.minkowski_p_norm == 1
        assert isinstance(pso.initializer, af.InitializerBall)
        assert pso.initializer.lower_limit == 0.2
        assert pso.initializer.upper_limit == 0.8
        assert pso.iterations_per_update == 10
        assert pso.number_of_cores == 2

        pso = af.PySwarmsLocal()

        assert pso.prior_passer.sigma == 3.0
        assert pso.prior_passer.use_errors == True
        assert pso.prior_passer.use_widths == True
        assert pso.n_particles == 50
        assert pso.iters == 2000
        assert pso.cognitive == 0.1
        assert pso.social == 0.2
        assert pso.inertia == 0.3
        assert pso.number_of_k_neighbors == 3
        assert pso.minkowski_p_norm == 2
        assert isinstance(pso.initializer, af.InitializerPrior)
        assert pso.iterations_per_update == 11
        assert pso.number_of_cores == 1

    def test__tag(self):

        pso = af.PySwarmsGlobal(
            n_particles=51, iters=2001, cognitive=0.4, social=0.5, inertia=0.6
        )

        assert pso.tag == "pyswarms_global[particles_51_c_0.4_s_0.5_i_0.6]"

        pso = af.PySwarmsLocal(
            n_particles=51, iters=2001, cognitive=0.4, social=0.5, inertia=0.6
        )

        assert pso.tag == "pyswarms_local[particles_51_c_0.4_s_0.5_i_0.6]"

    def test__samples_from_model(self):

        pyswarms = af.PySwarmsGlobal(paths=af.Paths())

        model = af.ModelMapper(mock_class=mock.MockClassx3)
        model.mock_class.one = af.LogUniformPrior(lower_limit=0.0, upper_limit=100.0)
        model.mock_class.two = af.LogUniformPrior(lower_limit=0.0, upper_limit=100.0)
        model.mock_class.three = af.LogUniformPrior(lower_limit=0.0, upper_limit=100.0)
        # model.mock_class.four = af.LogUniformPrior(lower_limit=0.0, upper_limit=100.0)

        samples = pyswarms.samples_via_sampler_from_model(model=model)

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
        assert samples.weights[0] == 1.0

        assert len(samples.parameters) == 500
        assert len(samples.log_likelihoods) == 500


class TestCopyWithNameExtension:
    @staticmethod
    def assert_non_linear_attributes_equal(copy):
        assert copy.paths.name == path.join("name", "one")

    def test__pyswarms(self):
        search = af.PySwarmsGlobal(af.Paths("name"))

        copy = search.copy_with_name_extension("one")
        self.assert_non_linear_attributes_equal(copy)
        assert isinstance(copy, af.PySwarmsGlobal)
        assert copy.prior_passer is search.prior_passer
        assert copy.n_particles is search.n_particles
        assert copy.iters is search.iters
        assert copy.cognitive == search.cognitive
        assert copy.social == search.social
        assert copy.inertia == search.inertia
        assert copy.ftol is search.ftol
        assert copy.initializer is search.initializer
        assert copy.iterations_per_update is search.iterations_per_update
        assert copy.number_of_cores is search.number_of_cores
