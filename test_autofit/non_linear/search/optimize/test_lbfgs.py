import pytest

import autofit as af

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

def test__loads_from_config_file_correct():
    search = af.LBFGS(
        tol=0.2,
        disp=True,
        maxcor=11,
        ftol=2.,
        gtol=3.,
        eps=4.,
        maxfun=25000,
        maxiter=26000,
        iprint=-2,
        maxls=21,
        initializer=af.InitializerBall(lower_limit=0.2, upper_limit=0.8),
        iterations_per_full_update=10,
        number_of_cores=2,
    )

    assert search.config_dict_search["tol"] == 0.2
    assert search.config_dict_options["maxcor"] == 11
    assert search.config_dict_options["ftol"] == 2.
    assert search.config_dict_options["gtol"] == 3.
    assert search.config_dict_options["eps"] == 4.
    assert search.config_dict_options["maxfun"] == 25000
    assert search.config_dict_options["maxiter"] == 26000
    assert search.config_dict_options["iprint"] == -2
    assert search.config_dict_options["maxls"] == 21
    assert search.config_dict_options["disp"] == True
    assert isinstance(search.initializer, af.InitializerBall)
    assert search.initializer.lower_limit == 0.2
    assert search.initializer.upper_limit == 0.8
    assert search.iterations_per_full_update == 10
    assert search.number_of_cores == 2

    search = af.LBFGS()

    assert search.config_dict_search["tol"] == None
    assert search.config_dict_options["maxcor"] == 10
    assert search.config_dict_options["ftol"] == 2.220446049250313e-09
    assert search.config_dict_options["gtol"] == 1e-05
    assert search.config_dict_options["eps"] == 1e-08
    assert search.config_dict_options["maxfun"] == 15000
    assert search.config_dict_options["maxiter"] == 15000
    assert search.config_dict_options["iprint"] == -1
    assert search.config_dict_options["maxls"] == 20
    assert search.config_dict_options["maxiter"] == 15000
    assert search.config_dict_options["disp"] == False
    assert isinstance(search.initializer, af.InitializerPrior)
    assert search.iterations_per_full_update == 1e99

