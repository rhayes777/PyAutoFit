import autofit as af
from autofit import database as m
from autofit.mock import mock
from autofit.non_linear.samples import Sample


def test_save_samples(
        paths,
        session
):
    paths.save_samples(
        af.OptimizerSamples(
            model=af.Model(
                mock.Gaussian
            ),
            samples=[Sample(
                log_likelihood=1.0,
                log_prior=1.0,
                weights=0.5,
                centre=1.0,
                intensity=2.0,
                sigma=3.0
            )]
        )
    )

    fit, = session.query(
        m.Fit
    ).all()

    assert fit.samples is not None
    

