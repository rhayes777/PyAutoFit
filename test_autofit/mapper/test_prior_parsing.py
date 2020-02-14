import autofit as af


def test_uniform():
    uniform_prior = af.Prior.from_dict(
        {
            "type": "Uniform",
            "lower_limit": 2,
            "upper_limit": 3
        }
    )

    assert isinstance(uniform_prior, af.UniformPrior)
    assert uniform_prior.lower_limit == 2
    assert uniform_prior.upper_limit == 3


def test_log_uniform():
    log_uniform_prior = af.Prior.from_dict(
        {
            "type": "LogUniform",
            "lower_limit": 0.2,
            "upper_limit": 0.3
        }
    )

    assert isinstance(log_uniform_prior, af.LogUniformPrior)
    assert log_uniform_prior.lower_limit == 0.2
    assert log_uniform_prior.upper_limit == 0.3


def test_gaussian():
    gaussian_prior = af.GaussianPrior.from_dict(
        {
            "type": "Gaussian",
            "lower_limit": -10,
            "upper_limit": 10,
            "mean": 3,
            "sigma": 4
        }
    )

    assert isinstance(gaussian_prior, af.GaussianPrior)
    assert gaussian_prior.lower_limit == -10
    assert gaussian_prior.upper_limit == 10
    assert gaussian_prior.mean == 3
    assert gaussian_prior.sigma == 4
