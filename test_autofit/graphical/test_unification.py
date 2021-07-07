import autofit as af
from autofit import graphical as g


def test():
    mean_field = g.MeanField({

    })

    print(mean_field.prior_count)
    mean_field.instance_for_arguments({})


def test_from_prior():
    prior = af.GaussianPrior(
        mean=1,
        sigma=2
    )
    message = g.AbstractMessage.from_prior(
        prior
    )

    assert message.id == prior.id
    assert message.mu == prior.mean
    assert message.sigma == prior.sigma
