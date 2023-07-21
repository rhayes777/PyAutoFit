from os import path

import pytest

import autofit as af

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def test__config__loads_from_file_correctly():

    search = af.Emcee(
        nwalkers=51,
        nsteps=2001,
        initializer=af.InitializerBall(lower_limit=0.2, upper_limit=0.8),
        auto_correlation_settings=af.AutoCorrelationsSettings(
            check_for_convergence=False,
            check_size=101,
            required_length=51,
            change_threshold=0.02
        ),
        number_of_cores=2,
    )

    assert search.config_dict_search["nwalkers"] == 51
    assert search.config_dict_run["nsteps"] == 2001
    assert isinstance(search.initializer, af.InitializerBall)
    assert search.initializer.lower_limit == 0.2
    assert search.initializer.upper_limit == 0.8
    assert search.auto_correlation_settings.check_for_convergence is False
    assert search.auto_correlation_settings.check_size == 101
    assert search.auto_correlation_settings.required_length == 51
    assert search.auto_correlation_settings.change_threshold == 0.02
    assert search.number_of_cores == 2

    search = af.Emcee()

    assert search.config_dict_search["nwalkers"] == 50
    assert search.config_dict_run["nsteps"] == 2000
    assert isinstance(search.initializer, af.InitializerPrior)
    assert search.auto_correlation_settings.check_for_convergence is True
    assert search.auto_correlation_settings.check_size == 100
    assert search.auto_correlation_settings.required_length == 50
    assert search.auto_correlation_settings.change_threshold == 0.01
    assert search.number_of_cores == 1

def test__samples_via_internal_from():

    search = af.Emcee()
    search.paths = af.DirectoryPaths(path_prefix=path.join("non_linear", "emcee"))
    search.paths._identifier = "tag"

    model = af.ModelMapper(mock_class=af.m.MockClassx4)
    model.mock_class.two = af.LogUniformPrior(lower_limit=1e-8, upper_limit=10.0)

    samples = search.samples_via_internal_from(model=model)

    assert isinstance(samples.parameter_lists, list)
    assert isinstance(samples.parameter_lists[0], list)
    assert isinstance(samples.log_likelihood_list, list)
    assert isinstance(samples.log_prior_list, list)
    assert isinstance(samples.log_posterior_list, list)
    assert isinstance(samples.weight_list, list)

    assert samples.parameter_lists[0] == pytest.approx(
        [0.009033, -0.057901, 10.192579, 0.480606], 1.0e-4
    )
    assert samples.log_likelihood_list[0] == pytest.approx(564.5526910632444, 1.0e-4)
    assert samples.log_prior_list[0] == pytest.approx(2.0807033929006, 1.0e-4)
    assert samples.weight_list[0] == pytest.approx(1.0, 1.0e-4)
    assert samples.total_steps == 1000
    assert samples.total_walkers == 10
    assert samples.auto_correlations.times[0] == pytest.approx(31.98507, 1.0e-4)


def test__median_pdf_parameters():
    search = af.Emcee()
    search.paths = af.DirectoryPaths(path_prefix=path.join("non_linear", "emcee"))
    search.paths._identifier = "tag"

    model = af.ModelMapper(mock_class=af.m.MockClassx4)
    model.mock_class.two = af.LogUniformPrior(lower_limit=1e-8, upper_limit=10.0)

    samples = search.samples_from(model=model)

    assert samples.median_pdf(as_instance=False) == pytest.approx(
        [0.008422, -0.026413, 9.9579656, 0.494618], 1.0e-3
    )

def test__vector_at_sigma__uses_output_files():
    search = af.Emcee()
    search.paths = af.DirectoryPaths(path_prefix=path.join("non_linear", "emcee"))
    search.paths._identifier = "tag"

    model = af.ModelMapper(mock_class=af.m.MockClassx4)
    model.mock_class.two = af.LogUniformPrior(lower_limit=1e-8, upper_limit=10.0)

    samples = search.samples_from(model=model)

    parameters = samples.values_at_sigma(sigma=3.0, as_instance=False)

    assert parameters[0][0:2] == pytest.approx((-0.003197, 0.019923), 1e-2)

    parameters = samples.values_at_sigma(sigma=1.0, as_instance=False)

    assert parameters[0][0:2] == pytest.approx((0.0042278, 0.01087681), 1e-2)

def test__autocorrelation_times():
    search = af.Emcee()
    search.paths = af.DirectoryPaths(path_prefix=path.join("non_linear", "emcee"))
    search.paths._identifier = "tag"

    model = af.ModelMapper(mock_class=af.m.MockClassx4)
    model.mock_class.two = af.LogUniformPrior(lower_limit=1e-8, upper_limit=10.0)

    samples = search.samples_from(model=model)

    assert samples.auto_correlations.previous_times == pytest.approx(
        [31.1079, 36.0910, 72.44768, 65.86194], 1.0e-4
    )
    assert samples.auto_correlations.times == pytest.approx(
        [31.98507, 36.51001, 73.47629, 67.67495], 1.0e-4
    )
