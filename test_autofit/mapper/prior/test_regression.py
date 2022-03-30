import autofit as af


def test_copy_limits():
    prior = af.GaussianPrior(
        mean=1,
        sigma=2,
        lower_limit=3,
        upper_limit=4
    )
    copied = prior.copy()
    assert prior.lower_limit == copied.lower_limit
    assert prior.upper_limit == copied.upper_limit
