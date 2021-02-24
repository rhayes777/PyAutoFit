import autofit as af


def test_embedded_priors():
    prior_model = af.PriorModel(
        af.GaussianPrior,
        mean=af.GaussianPrior(
            mean=0,
            sigma=1
        ),
        sigma=af.GaussianPrior(
            mean=4,
            sigma=1
        )
    )

    assert isinstance(
        prior_model.random_instance().value_for(0.5),
        float
    )
