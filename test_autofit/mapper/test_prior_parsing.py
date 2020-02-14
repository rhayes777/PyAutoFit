import autofit as af


def test_prior_from_dict():
    uniform_prior = af.Prior.from_dict(
        {
            "type": "Uniform"
        }
    )

    assert isinstance(uniform_prior, af.UniformPrior)

    log_uniform_prior = af.Prior.from_dict(
        {
            "type": "LogUniform"
        }
    )

    assert isinstance(log_uniform_prior, af.UniformPrior)

    gaussian_prior = af.GaussianPrior.from_dict(
        {
            "type": "Gaussian",
            "mean": 3,
            "sigma": 4
        }
    )

    assert isinstance(gaussian_prior, af.GaussianPrior)
