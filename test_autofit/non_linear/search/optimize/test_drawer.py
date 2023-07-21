from os import path

import pytest

import autofit as af

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def test__loads_from_config_file_correct():
    search = af.Drawer(
        total_draws=5,
        initializer=af.InitializerBall(lower_limit=0.2, upper_limit=0.8),
    )

    assert search.config_dict_search["total_draws"] == 5
    assert isinstance(search.initializer, af.InitializerBall)
    assert search.initializer.lower_limit == 0.2
    assert search.initializer.upper_limit == 0.8
    assert search.number_of_cores == 1

    search = af.Drawer()

    assert search.config_dict_search["total_draws"] == 10
    assert isinstance(search.initializer, af.InitializerPrior)
