from os import path

import pytest

import autofit as af

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


class TestEmceeConfig:

    def test__loads_from_config_file_correct(self):

        emcee = af.Emcee(
            prior_passer=af.PriorPasser(sigma=2.0, use_errors=False, use_widths=False),
            nwalkers=51,
            nsteps=2001,
            initializer=af.InitializerBall(lower_limit=0.2, upper_limit=0.8),
            auto_correlations_settings=af.AutoCorrelationsSettings(
                check_for_convergence=False,
                check_size=101,
                required_length=51,
                change_threshold=0.02
            ),
            number_of_cores=2,
        )

        assert emcee.prior_passer.sigma == 2.0
        assert emcee.prior_passer.use_errors is False
        assert emcee.prior_passer.use_widths is False
        assert emcee.config_dict_search["nwalkers"] == 51
        assert emcee.config_dict_run["nsteps"] == 2001
        assert isinstance(emcee.initializer, af.InitializerBall)
        assert emcee.initializer.lower_limit == 0.2
        assert emcee.initializer.upper_limit == 0.8
        assert emcee.auto_correlations_settings.check_for_convergence is False
        assert emcee.auto_correlations_settings.check_size == 101
        assert emcee.auto_correlations_settings.required_length == 51
        assert emcee.auto_correlations_settings.change_threshold == 0.02
        assert emcee.number_of_cores == 2

        emcee = af.Emcee()

        assert emcee.prior_passer.sigma == 3.0
        assert emcee.prior_passer.use_errors is True
        assert emcee.prior_passer.use_widths is True
        assert emcee.config_dict_search["nwalkers"] == 50
        assert emcee.config_dict_run["nsteps"] == 2000
        assert isinstance(emcee.initializer, af.InitializerPrior)
        assert emcee.auto_correlations_settings.check_for_convergence is True
        assert emcee.auto_correlations_settings.check_size == 100
        assert emcee.auto_correlations_settings.required_length == 50
        assert emcee.auto_correlations_settings.change_threshold == 0.01
        assert emcee.number_of_cores == 1

    def test__samples_from_model(self):

        emcee = af.Emcee()
        emcee.paths = af.DirectoryPaths(path_prefix=path.join("non_linear", "emcee"))
        emcee.paths._identifier = "tag"

        model = af.ModelMapper(mock_class=af.m.MockClassx4)
        model.mock_class.two = af.LogUniformPrior(lower_limit=1e-8, upper_limit=10.0)

        samples = emcee.samples_from(model=model)

        assert isinstance(samples.parameter_lists, list)
        assert isinstance(samples.parameter_lists[0], list)
        assert isinstance(samples.log_likelihood_list, list)
        assert isinstance(samples.log_prior_list, list)
        assert isinstance(samples.log_posterior_list, list)
        assert isinstance(samples.weight_list, list)

        assert samples.parameter_lists[0] == pytest.approx(
            [0.173670, 0.162607, 3095.28, 0.62104], 1.0e-4
        )
        assert samples.log_likelihood_list[0] == pytest.approx(-17257775239.32677, 1.0e-4)
        assert samples.log_prior_list[0] == pytest.approx(1.6102016075510708, 1.0e-4)
        assert samples.weight_list[0] == pytest.approx(1.0, 1.0e-4)
        assert samples.total_steps == 1000
        assert samples.total_walkers == 10
        assert samples.auto_correlations.times[0] == pytest.approx(31.98507, 1.0e-4)


class TestEmceeOutput:
    def test__median_pdf_parameters(self):
        emcee = af.Emcee()
        emcee.paths = af.DirectoryPaths(path_prefix=path.join("non_linear", "emcee"))
        emcee.paths._identifier = "tag"

        model = af.ModelMapper(mock_class=af.m.MockClassx4)
        model.mock_class.two = af.LogUniformPrior(lower_limit=1e-8, upper_limit=10.0)

        samples = emcee.samples_from(model=model)

        assert samples.median_pdf_vector == pytest.approx(
            [0.008422, -0.026413, 9.9579656, 0.494618], 1.0e-3
        )

    def test__vector_at_sigma__uses_output_files(self):
        emcee = af.Emcee()
        emcee.paths = af.DirectoryPaths(path_prefix=path.join("non_linear", "emcee"))
        emcee.paths._identifier = "tag"

        model = af.ModelMapper(mock_class=af.m.MockClassx4)
        model.mock_class.two = af.LogUniformPrior(lower_limit=1e-8, upper_limit=10.0)

        samples = emcee.samples_from(model=model)

        parameters = samples.vector_at_sigma(sigma=3.0)

        assert parameters[0][0:2] == pytest.approx((-0.003197, 0.019923), 1e-2)

        parameters = samples.vector_at_sigma(sigma=1.0)

        assert parameters[0][0:2] == pytest.approx((0.0042278, 0.01087681), 1e-2)

    def test__autocorrelation_times(self):
        emcee = af.Emcee()
        emcee.paths = af.DirectoryPaths(path_prefix=path.join("non_linear", "emcee"))
        emcee.paths._identifier = "tag"

        model = af.ModelMapper(mock_class=af.m.MockClassx4)
        model.mock_class.two = af.LogUniformPrior(lower_limit=1e-8, upper_limit=10.0)

        samples = emcee.samples_from(model=model)

        assert samples.auto_correlations.previous_times == pytest.approx(
            [31.1079, 36.0910, 72.44768, 65.86194], 1.0e-4
        )
        assert samples.auto_correlations.times == pytest.approx(
            [31.98507, 36.51001, 73.47629, 67.67495], 1.0e-4
        )
