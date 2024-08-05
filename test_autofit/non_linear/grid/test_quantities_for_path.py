import pytest

import autofit as af


@pytest.fixture(name="result")
def make_result():
    model = af.Model(
        af.Gaussian,
    )
    grid_priors = [model.centre, model.normalization]
    lower_limits_lists = [
        [0.0, 1.0],
        [0.0, 2.0],
        [2.0, 1.0],
        [2.0, 2.0],
    ]

    sample = af.Sample(
        1.0,
        1.0,
        1.0,
        {
            "centre": 1.0,
            "normalization": 2.0,
            "sigma": 3.0,
        },
    )

    def make_samples(centre, normalization):
        return af.Samples(
            model=af.Model(
                af.Gaussian,
                centre=af.UniformPrior(
                    lower_limit=centre,
                    upper_limit=centre + 2.0,
                ),
                normalization=af.UniformPrior(
                    lower_limit=normalization,
                    upper_limit=normalization + 1.0,
                ),
            ),
            sample_list=[sample],
        )

    samples = [
        make_samples(centre, normalisation)
        for centre, normalisation in lower_limits_lists
    ]

    return af.GridSearchResult(
        samples=samples,
        lower_limits_lists=lower_limits_lists,
        grid_priors=grid_priors,
    )


@pytest.mark.parametrize(
    "name, expected",
    [
        ("centre", [1.0, 1.0, 3.0, 3.0]),
        ("normalization", [1.5, 2.5, 1.5, 2.5]),
    ],
)
def test_physical_centres_from(result, name, expected):
    assert result.physical_centres_lists_from(name) == expected
    assert result.shape == (2, 2)


def test_two_physical_centres(result):
    assert result.physical_centres_lists_from(("centre", "normalization")) == [
        (1.0, 1.5),
        (1.0, 2.5),
        (3.0, 1.5),
        (3.0, 2.5),
    ]
