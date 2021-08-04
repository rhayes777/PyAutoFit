import pytest

import autofit as af


@pytest.mark.parametrize(
    "upper_limit, physical_value",
    [
        (1.0, 0.5),
        (2.0, 1.0),
        (4.0, 2.0),
    ]
)
def test_physical_lower_limits(
        upper_limit,
        physical_value
):
    model = af.Model(
        af.Gaussian,
        centre=af.UniformPrior(
            lower_limit=0.0,
            upper_limit=upper_limit
        )
    )
    result = af.GridSearchResult(
        results=[],
        lower_limit_lists=[
            [0.5]
        ],
        grid_priors=[model.centre]
    )

    assert result.physical_lower_limit_lists == [[physical_value]]
