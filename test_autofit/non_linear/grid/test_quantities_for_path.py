import autofit as af


def test_physical_centres_from():
    model = af.Model(
        af.Gaussian,
    )
    grid_priors = [model.centre]
    lower_limits_lists = [[0.0], [2.0]]

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

    samples = [
        af.Samples(
            model=af.Model(
                af.Gaussian,
                centre=af.UniformPrior(
                    lower_limit=0.0,
                    upper_limit=2.0,
                ),
            ),
            sample_list=[sample],
        ),
        af.Samples(
            model=af.Model(
                af.Gaussian,
                centre=af.UniformPrior(
                    lower_limit=2.0,
                    upper_limit=4.0,
                ),
            ),
            sample_list=[sample],
        ),
    ]

    result = af.GridSearchResult(
        samples=samples,
        lower_limits_lists=lower_limits_lists,
        grid_priors=grid_priors,
    )

    assert result.physical_centres_lists_from("centre") == [[1.0], [3.0]]
