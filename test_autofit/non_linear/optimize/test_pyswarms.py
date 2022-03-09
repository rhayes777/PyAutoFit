from os import path

import pytest

import autofit as af

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


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
        assert pso.prior_passer.use_errors is False
        assert pso.prior_passer.use_widths is False
        assert pso.config_dict_search["n_particles"] == 51
        assert pso.config_dict_search["cognitive"] == 0.4
        assert pso.config_dict_run["iters"] == 2001
        assert isinstance(pso.initializer, af.InitializerBall)
        assert pso.initializer.lower_limit == 0.2
        assert pso.initializer.upper_limit == 0.8
        assert pso.iterations_per_update == 10
        assert pso.number_of_cores == 2

        pso = af.PySwarmsGlobal()

        assert pso.prior_passer.sigma == 3.0
        assert pso.prior_passer.use_errors is True
        assert pso.prior_passer.use_widths is True
        assert pso.config_dict_search["n_particles"] == 50
        assert pso.config_dict_search["cognitive"] == 0.1
        assert pso.config_dict_run["iters"] == 2000
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
        assert pso.prior_passer.use_errors is False
        assert pso.prior_passer.use_widths is False
        assert pso.config_dict_search["n_particles"] == 51
        assert pso.config_dict_search["cognitive"] == 0.4
        assert pso.config_dict_run["iters"] == 2001
        assert isinstance(pso.initializer, af.InitializerBall)
        assert pso.initializer.lower_limit == 0.2
        assert pso.initializer.upper_limit == 0.8
        assert pso.iterations_per_update == 10
        assert pso.number_of_cores == 2

        pso = af.PySwarmsLocal()

        assert pso.prior_passer.sigma == 3.0
        assert pso.prior_passer.use_errors is True
        assert pso.prior_passer.use_widths is True
        assert pso.config_dict_search["n_particles"] == 50
        assert pso.config_dict_search["cognitive"] == 0.1
        assert pso.config_dict_run["iters"] == 2000
        assert isinstance(pso.initializer, af.InitializerPrior)
        assert pso.iterations_per_update == 11
        assert pso.number_of_cores == 1

    def test__samples_from_model(self):
        pyswarms = af.PySwarmsGlobal()
        pyswarms.paths = af.DirectoryPaths(path_prefix=path.join("non_linear", "pyswarms"))
        pyswarms.paths._identifier = "tag"

        model = af.ModelMapper(mock_class=af.m.MockClassx3)
        model.mock_class.one = af.LogUniformPrior(lower_limit=1e-8, upper_limit=100.0)
        model.mock_class.two = af.LogUniformPrior(lower_limit=1e-8, upper_limit=100.0)
        model.mock_class.three = af.LogUniformPrior(lower_limit=1e-8, upper_limit=100.0)
        # model.mock_class.four = af.LogUniformPrior(lower_limit=1e-8, upper_limit=100.0)

        samples = pyswarms.samples_from(model=model)

        assert isinstance(samples.parameter_lists, list)
        assert isinstance(samples.parameter_lists[0], list)
        assert isinstance(samples.log_likelihood_list, list)
        assert isinstance(samples.log_prior_list, list)
        assert isinstance(samples.log_posterior_list, list)

        assert samples.parameter_lists[0] == pytest.approx(
            [50.1254, 1.04626, 10.09456], 1.0e-4
        )

        assert samples.log_likelihood_list[0] == pytest.approx(-5071.80777, 1.0e-4)
        assert samples.log_posterior_list[0] == pytest.approx(-5070.73298, 1.0e-4)
        assert samples.weight_list[0] == 1.0

        assert len(samples.parameter_lists) == 500
        assert len(samples.log_likelihood_list) == 500
