import pytest

import autofit as af
from autofit.tools.util import to_dict


@pytest.fixture(name="dynesty_dict")
def make_dynesty_dict():
    return {
        "arguments": {
            "iterations_per_update": 500,
            "name": "",
            "number_of_cores": 1,
            "path_prefix": "",
            "prior_passer": {
                "arguments": {"sigma": 3.0, "use_errors": True, "use_widths": True},
                "type": "autofit.non_linear.abstract_search.PriorPasser",
            },
            "unique_tag": None,
        },
        "type": "autofit.non_linear.nest.dynesty.static.DynestyStatic",
    }


def test_dict(dynesty_dict):
    dynesty = af.DynestyStatic()
    assert to_dict(dynesty) == dynesty_dict


def test_prior_passer():
    prior_passer = af.PriorPasser(
        sigma=1.0,
        use_errors=False,
        use_widths=False,
    )
    assert to_dict(prior_passer) == {
        "arguments": {
            "sigma": 1.0,
            "use_errors": False,
            "use_widths": False,
        },
        "type": "autofit.non_linear.abstract_search.PriorPasser",
    }


def test_initializer():
    initializer = af.InitializerBall(lower_limit=0.0, upper_limit=1.0)
    assert to_dict(initializer) == {
        "arguments": {"lower_limit": 0.0, "upper_limit": 1.0},
        "type": "autofit.non_linear.initializer.InitializerBall",
    }
