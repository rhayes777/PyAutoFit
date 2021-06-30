from os import path

import pytest

import autofit as af
from autofit.mock import mock

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


class TestLBFGSConfig:
    def test__loads_from_config_file_correct(self):
        lbfgs = af.LBFGS(
            prior_passer=af.PriorPasser(sigma=2.0, use_errors=False, use_widths=False),
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
            iterations_per_update=10,
            number_of_cores=2,
        )

        assert lbfgs.prior_passer.sigma == 2.0
        assert lbfgs.prior_passer.use_errors is False
        assert lbfgs.prior_passer.use_widths is False
        assert lbfgs.config_dict_search["tol"] == 0.2
        assert lbfgs.config_dict_search["disp"] == True
        assert lbfgs.config_dict_search["maxcor"] == 11
        assert lbfgs.config_dict_search["ftol"] == 2.
        assert lbfgs.config_dict_search["gtol"] == 3.
        assert lbfgs.config_dict_search["eps"] == 4.
        assert lbfgs.config_dict_search["maxfun"] == 25000
        assert lbfgs.config_dict_search["maxiter"] == 26000
        assert lbfgs.config_dict_search["iprint"] == -2
        assert lbfgs.config_dict_search["maxls"] == 21
        assert lbfgs.config_dict_options["maxiter"] == None
        assert lbfgs.config_dict_options["disp"] == False
        assert isinstance(lbfgs.initializer, af.InitializerBall)
        assert lbfgs.initializer.lower_limit == 0.2
        assert lbfgs.initializer.upper_limit == 0.8
        assert lbfgs.iterations_per_update == 10
        assert lbfgs.number_of_cores == 2

        lbfgs = af.LBFGS()

        assert lbfgs.prior_passer.sigma == 3.0
        assert lbfgs.prior_passer.use_errors is True
        assert lbfgs.prior_passer.use_widths is True
        assert lbfgs.config_dict_search["tol"] == None
        assert lbfgs.config_dict_search["disp"] == None
        assert lbfgs.config_dict_search["maxcor"] == 10
        assert lbfgs.config_dict_search["ftol"] == 2.220446049250313e-09
        assert lbfgs.config_dict_search["gtol"] == 1e-05
        assert lbfgs.config_dict_search["eps"] == 1e-08
        assert lbfgs.config_dict_search["maxfun"] == 15000
        assert lbfgs.config_dict_search["maxiter"] == 15000
        assert lbfgs.config_dict_search["iprint"] == -1
        assert lbfgs.config_dict_search["maxls"] == 20
        assert lbfgs.config_dict_options["maxiter"] == None
        assert lbfgs.config_dict_options["disp"] == False
        assert isinstance(lbfgs.initializer, af.InitializerPrior)
        assert lbfgs.iterations_per_update == 11

    def test__samples_from_model(self):
        
        LBFGS = af.LBFGS()
        LBFGS.paths = af.DirectoryPaths(path_prefix=path.join("non_linear", "LBFGS"))
        LBFGS.paths._identifier = "tag"

        model = af.ModelMapper(mock_class=mock.MockClassx3)
        model.mock_class.one = af.LogUniformPrior(lower_limit=1e-8, upper_limit=100.0)
        model.mock_class.two = af.LogUniformPrior(lower_limit=1e-8, upper_limit=100.0)
        model.mock_class.three = af.LogUniformPrior(lower_limit=1e-8, upper_limit=100.0)
        # model.mock_class.four = af.LogUniformPrior(lower_limit=1e-8, upper_limit=100.0)

        samples = LBFGS.samples_from(model=model)

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
