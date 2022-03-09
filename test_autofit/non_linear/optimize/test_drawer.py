from os import path

import pytest

import autofit as af

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


class TestDrawerConfig:
    def test__loads_from_config_file_correct(self):
        drawer = af.Drawer(
            total_draws=5,
            prior_passer=af.PriorPasser(sigma=2.0, use_errors=False, use_widths=False),
            initializer=af.InitializerBall(lower_limit=0.2, upper_limit=0.8),
        )

        assert drawer.prior_passer.sigma == 2.0
        assert drawer.prior_passer.use_errors is False
        assert drawer.prior_passer.use_widths is False
        assert drawer.config_dict_search["total_draws"] == 5
        assert isinstance(drawer.initializer, af.InitializerBall)
        assert drawer.initializer.lower_limit == 0.2
        assert drawer.initializer.upper_limit == 0.8
        assert drawer.number_of_cores == 1

        drawer = af.Drawer()

        assert drawer.prior_passer.sigma == 3.0
        assert drawer.prior_passer.use_errors is True
        assert drawer.prior_passer.use_widths is True
        assert drawer.config_dict_search["total_draws"] == 10
        assert isinstance(drawer.initializer, af.InitializerPrior)

    def test__samples_from_model(self):
        
        drawer = af.Drawer()
        drawer.paths = af.DirectoryPaths(path_prefix=path.join("non_linear", "drawer"))
        drawer.paths._identifier = "tag"

        model = af.ModelMapper(mock_class=af.m.MockClassx3)
        model.mock_class.one = af.LogUniformPrior(lower_limit=1e-8, upper_limit=100.0)
        model.mock_class.two = af.LogUniformPrior(lower_limit=1e-8, upper_limit=100.0)
        model.mock_class.three = af.LogUniformPrior(lower_limit=1e-8, upper_limit=100.0)

        samples = drawer.samples_from(model=model)

        assert isinstance(samples.parameter_lists, list)
        assert isinstance(samples.parameter_lists[0], list)
        assert isinstance(samples.log_likelihood_list, list)
        assert isinstance(samples.log_prior_list, list)
        assert isinstance(samples.log_posterior_list, list)

        assert samples.parameter_lists[0] == pytest.approx(
            [49.507679, 49.177471, 14.76753], 1.0e-4
        )

        assert samples.log_likelihood_list[0] == pytest.approx(-2763.925766, 1.0e-4)
        assert samples.log_posterior_list[0] == pytest.approx(-2763.817517, 1.0e-4)
        assert samples.weight_list[0] == 1.0

        assert len(samples.parameter_lists) == 3
        assert len(samples.log_likelihood_list) == 3
