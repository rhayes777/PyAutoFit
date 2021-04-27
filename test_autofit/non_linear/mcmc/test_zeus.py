import pytest
import autofit as af

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


class TestZeusConfig:

    def test__loads_from_config_file_correct(self):

        zeus = af.Zeus(
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
            tune=False,
            number_of_cores=2,
        )

        assert zeus.prior_passer.sigma == 2.0
        assert zeus.prior_passer.use_errors is False
        assert zeus.prior_passer.use_widths is False
        assert zeus.config_dict_search["nwalkers"] == 51
        assert zeus.config_dict_run["nsteps"] == 2001
        assert zeus.config_dict_run["tune"] == False
        assert isinstance(zeus.initializer, af.InitializerBall)
        assert zeus.initializer.lower_limit == 0.2
        assert zeus.initializer.upper_limit == 0.8
        assert zeus.auto_correlations_settings.check_for_convergence is False
        assert zeus.auto_correlations_settings.check_size == 101
        assert zeus.auto_correlations_settings.required_length == 51
        assert zeus.auto_correlations_settings.change_threshold == 0.02
        assert zeus.number_of_cores == 2

        zeus = af.Zeus()

        assert zeus.prior_passer.sigma == 3.0
        assert zeus.prior_passer.use_errors is True
        assert zeus.prior_passer.use_widths is True
        assert zeus.config_dict_search["nwalkers"] == 50
        assert zeus.config_dict_run["nsteps"] == 2000
        assert zeus.config_dict_run["tune"] == True
        assert isinstance(zeus.initializer, af.InitializerPrior)
        assert zeus.auto_correlations_settings.check_for_convergence is True
        assert zeus.auto_correlations_settings.check_size == 100
        assert zeus.auto_correlations_settings.required_length == 50
        assert zeus.auto_correlations_settings.change_threshold == 0.01
        assert zeus.number_of_cores == 1

#     def test__samples_from_model(self):
#
#         zeus = af.Zeus()
#         zeus.paths = af.DirectoryPaths(path_prefix=path.join("non_linear", "zeus"))
#         zeus.paths._identifier = "tag"
#
#         model = af.ModelMapper(mock_class=mock.MockClassx4)
#         model.mock_class.two = af.LogUniformPrior(lower_limit=1e-8, upper_limit=10.0)
#
#         fitness_function = zeus.fitness_function_from_model_and_analysis(
#             model=model, analysis=None,
#         )
#
#         zeus_sampler = ze.EnsembleSampler(nwalkers=100, ndim=4, logprob_fn=fitness_function)
#
#         samples = zeus.samples_via_sampler_from_model(model=model, zeus_sampler=zeus_sampler)
#
#         assert isinstance(samples.parameters, list)
#         assert isinstance(samples.parameters[0], list)
#         assert isinstance(samples.log_likelihoods, list)
#         assert isinstance(samples.log_priors, list)
#         assert isinstance(samples.log_posteriors, list)
#         assert isinstance(samples.weights, list)
#
#         assert samples.parameters[0] == pytest.approx(
#             [0.173670, 0.162607, 3095.28, 0.62104], 1.0e-4
#         )
#         assert samples.log_likelihoods[0] == pytest.approx(-17257775239.32677, 1.0e-4)
#         assert samples.log_priors[0] == pytest.approx(1.6102016075510708, 1.0e-4)
#         assert samples.weights[0] == pytest.approx(1.0, 1.0e-4)
#         assert samples.total_steps == 1000
#         assert samples.total_walkers == 10
#         assert samples.auto_correlations.times[0] == pytest.approx(31.98507, 1.0e-4)
#
#
# class TestZeusOutput:
#     def test__median_pdf_parameters(self):
#         zeus = af.Zeus()
#         zeus.paths = af.DirectoryPaths(path_prefix=path.join("non_linear", "zeus"))
#         zeus.paths._identifier = "tag"
#
#         model = af.ModelMapper(mock_class=mock.MockClassx4)
#         model.mock_class.two = af.LogUniformPrior(lower_limit=1e-8, upper_limit=10.0)
#
#         samples = zeus.samples_via_sampler_from_model(model=model)
#
#         assert samples.median_pdf_vector == pytest.approx(
#             [0.008422, -0.026413, 9.9579656, 0.494618], 1.0e-3
#         )
#
#     def test__vector_at_sigma__uses_output_files(self):
#         zeus = af.Zeus()
#         zeus.paths = af.DirectoryPaths(path_prefix=path.join("non_linear", "zeus"))
#         zeus.paths._identifier = "tag"
#
#         model = af.ModelMapper(mock_class=mock.MockClassx4)
#         model.mock_class.two = af.LogUniformPrior(lower_limit=1e-8, upper_limit=10.0)
#
#         samples = zeus.samples_via_sampler_from_model(model=model)
#
#         parameters = samples.vector_at_sigma(sigma=3.0)
#
#         assert parameters[0][0:2] == pytest.approx((-0.003197, 0.019923), 1e-2)
#
#         parameters = samples.vector_at_sigma(sigma=1.0)
#
#         assert parameters[0][0:2] == pytest.approx((0.0042278, 0.01087681), 1e-2)
#
#     def test__autocorrelation_times(self):
#         zeus = af.Zeus()
#         zeus.paths = af.DirectoryPaths(path_prefix=path.join("non_linear", "zeus"))
#         zeus.paths._identifier = "tag"
#
#         model = af.ModelMapper(mock_class=mock.MockClassx4)
#         model.mock_class.two = af.LogUniformPrior(lower_limit=1e-8, upper_limit=10.0)
#
#         samples = zeus.samples_via_sampler_from_model(model=model)
#
#         assert samples.auto_correlations.previous_times == pytest.approx(
#             [31.1079, 36.0910, 72.44768, 65.86194], 1.0e-4
#         )
#         assert samples.auto_correlations.times == pytest.approx(
#             [31.98507, 36.51001, 73.47629, 67.67495], 1.0e-4
#         )
