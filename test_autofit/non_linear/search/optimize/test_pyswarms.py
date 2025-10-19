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
        iterations_per_full_update=10,
        number_of_cores=2,
    )

    assert search.config_dict_search["n_particles"] == 51
    assert search.config_dict_search["cognitive"] == 0.4
    assert search.config_dict_run["iters"] == 2001
    assert isinstance(search.initializer, af.InitializerBall)
    assert search.initializer.lower_limit == 0.2
    assert search.initializer.upper_limit == 0.8
    assert search.iterations_per_full_update == 10
    assert search.number_of_cores == 2

    search = af.PySwarmsGlobal()

    assert search.config_dict_search["n_particles"] == 50
    assert search.config_dict_search["cognitive"] == 0.1
    assert search.config_dict_run["iters"] == 2000
    assert isinstance(search.initializer, af.InitializerPrior)
    assert search.iterations_per_full_update == 11
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
        iterations_per_full_update=10,
        number_of_cores=2,
    )

    assert search.config_dict_search["n_particles"] == 51
    assert search.config_dict_search["cognitive"] == 0.4
    assert search.config_dict_run["iters"] == 2001
    assert isinstance(search.initializer, af.InitializerBall)
    assert search.initializer.lower_limit == 0.2
    assert search.initializer.upper_limit == 0.8
    assert search.iterations_per_full_update == 10
    assert search.number_of_cores == 2

    search = af.PySwarmsLocal()

    assert search.config_dict_search["n_particles"] == 50
    assert search.config_dict_search["cognitive"] == 0.1
    assert search.config_dict_run["iters"] == 2000
    assert isinstance(search.initializer, af.InitializerPrior)
    assert search.iterations_per_full_update == 11
    assert search.number_of_cores == 1



