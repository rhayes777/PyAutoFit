from os import path

import pytest

import autofit as af

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def test__loads_from_config_file_correct():
    search = af.PySwarmsGlobal(
        n_particles=51,
        iters=2001,
        cognitive=0.4,
        social=0.5,
        inertia=0.6,
        initializer=af.InitializerBall(lower_limit=0.2, upper_limit=0.8),
        iterations_per_update=10,
        number_of_cores=2,
    )

    assert search.config_dict_search["n_particles"] == 51
    assert search.config_dict_search["cognitive"] == 0.4
    assert search.config_dict_run["iters"] == 2001
    assert isinstance(search.initializer, af.InitializerBall)
    assert search.initializer.lower_limit == 0.2
    assert search.initializer.upper_limit == 0.8
    assert search.iterations_per_update == 10
    assert search.number_of_cores == 2

    search = af.PySwarmsGlobal()

    assert search.config_dict_search["n_particles"] == 50
    assert search.config_dict_search["cognitive"] == 0.1
    assert search.config_dict_run["iters"] == 2000
    assert isinstance(search.initializer, af.InitializerPrior)
    assert search.iterations_per_update == 11
    assert search.number_of_cores == 1

    search = af.PySwarmsLocal(
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

    assert search.config_dict_search["n_particles"] == 51
    assert search.config_dict_search["cognitive"] == 0.4
    assert search.config_dict_run["iters"] == 2001
    assert isinstance(search.initializer, af.InitializerBall)
    assert search.initializer.lower_limit == 0.2
    assert search.initializer.upper_limit == 0.8
    assert search.iterations_per_update == 10
    assert search.number_of_cores == 2

    search = af.PySwarmsLocal()

    assert search.config_dict_search["n_particles"] == 50
    assert search.config_dict_search["cognitive"] == 0.1
    assert search.config_dict_run["iters"] == 2000
    assert isinstance(search.initializer, af.InitializerPrior)
    assert search.iterations_per_update == 11
    assert search.number_of_cores == 1


def test__samples_via_internal_from():
    search = af.PySwarmsGlobal()
    search.paths = af.DirectoryPaths(path_prefix=path.join("non_linear", "pyswarms"))
    search.paths._identifier = "tag"

    model = af.ModelMapper(mock_class=af.m.MockClassx3)
    model.mock_class.one = af.LogUniformPrior(lower_limit=1e-8, upper_limit=100.0)
    model.mock_class.two = af.LogUniformPrior(lower_limit=1e-8, upper_limit=100.0)
    model.mock_class.three = af.LogUniformPrior(lower_limit=1e-8, upper_limit=100.0)
    # model.mock_class.four = af.LogUniformPrior(lower_limit=1e-8, upper_limit=100.0)

    samples = search.samples_via_internal_from(model=model)

    assert isinstance(samples.parameter_lists, list)
    assert isinstance(samples.parameter_lists[0], list)
    assert isinstance(samples.log_likelihood_list, list)
    assert isinstance(samples.log_prior_list, list)
    assert isinstance(samples.log_posterior_list, list)

    assert samples.parameter_lists[0] == pytest.approx(
        [50.1254, 1.04626, 10.09456], 1.0e-4
    )

    assert samples.log_likelihood_list[0] == pytest.approx(-2780.995417544426, 1.0e-4)
    assert samples.log_posterior_list[0] == pytest.approx(-2779.9206, 1.0e-4)
    assert samples.weight_list[0] == 1.0

    assert len(samples.parameter_lists) == 2
    assert len(samples.log_likelihood_list) == 2
