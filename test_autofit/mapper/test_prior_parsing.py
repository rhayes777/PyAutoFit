import autofit as af


def test_prior_from_dict():
    uniform_prior = af.Prior.from_dict(
        {
            "type": "Uniform"
        }
    )

    assert isinstance(uniform_prior, af.UniformPrior)
