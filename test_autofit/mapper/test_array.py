import autofit as af


def test_instantiate():
    array = af.Array(
        shape=(2, 2),
        prior=af.GaussianPrior(mean=0.0, sigma=1.0),
    )
    assert array.prior_count == 4
