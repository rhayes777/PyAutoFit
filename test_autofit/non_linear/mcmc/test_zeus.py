import pytest
import autofit as af

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def test__loads_from_config_file_correct():

    zeus = af.Zeus(
        nwalkers=51,
        nsteps=2001,
        initializer=af.InitializerBall(lower_limit=0.2, upper_limit=0.8),
        auto_correlation_settings=af.AutoCorrelationsSettings(
            check_for_convergence=False,
            check_size=101,
            required_length=51,
            change_threshold=0.02
        ),
        tune=False,
        number_of_cores=2,
    )

    assert zeus.config_dict_search["nwalkers"] == 51
    assert zeus.config_dict_run["nsteps"] == 2001
    assert zeus.config_dict_run["tune"] == False
    assert isinstance(zeus.initializer, af.InitializerBall)
    assert zeus.initializer.lower_limit == 0.2
    assert zeus.initializer.upper_limit == 0.8
    assert zeus.auto_correlation_settings.check_for_convergence is False
    assert zeus.auto_correlation_settings.check_size == 101
    assert zeus.auto_correlation_settings.required_length == 51
    assert zeus.auto_correlation_settings.change_threshold == 0.02
    assert zeus.number_of_cores == 2

    zeus = af.Zeus()

    assert zeus.config_dict_search["nwalkers"] == 50
    assert zeus.config_dict_run["nsteps"] == 2000
    assert zeus.config_dict_run["tune"] == True
    assert isinstance(zeus.initializer, af.InitializerPrior)
    assert zeus.auto_correlation_settings.check_for_convergence is True
    assert zeus.auto_correlation_settings.check_size == 100
    assert zeus.auto_correlation_settings.required_length == 50
    assert zeus.auto_correlation_settings.change_threshold == 0.01
    assert zeus.number_of_cores == 1

