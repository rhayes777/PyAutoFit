import numpy as np

import autofit as af
from autofit import message_passing as mp
from .model import Gaussian, make_data, _gaussian, _likelihood

n_observations = 100

prior = af.GaussianPrior(
    mean=0,
    sigma=40
)


def test_gaussian():
    x, y = make_data(
        Gaussian(
            centre=50.0,
            intensity=25.0,
            sigma=10.0
        ),
        n_observations
    )

    observations = mp.Plate(
        name="observations"
    )

    # TODO: Can we derive variables by looking at function argument names?
    x_ = mp.Variable(
        "x", observations
    )
    y_ = mp.Variable(
        "y", observations
    )
    z = mp.Variable(
        "z", observations
    )
    centre = mp.Variable(
        "centre"
    )
    intensity = mp.Variable(
        "intensity"
    )
    sigma = mp.Variable(
        "sigma"
    )

    gaussian = mp.Factor(
        _gaussian
    )(
        x_,
        centre,
        intensity,
        sigma
    ) == z
    likelihood = mp.Factor(
        _likelihood
    )(z, y_)

    # TODO: Can priors look like autofit priors? Could mp objects derive promise functionality from autofit?
    prior_centre = mp.Factor(
        prior
    )(centre)
    prior_intensity = mp.Factor(
        prior
    )(intensity)
    prior_sigma = mp.Factor(
        prior
    )(sigma)

    model = likelihood * gaussian * prior_centre * prior_sigma * prior_intensity

    model_approx = mp.MeanFieldApproximation.from_kws(
        model,
        centre=mp.NormalMessage.from_prior(
            prior
        ),
        intensity=mp.NormalMessage.from_prior(
            prior
        ),
        sigma=mp.NormalMessage.from_prior(
            prior
        ),
        x=mp.FixedMessage(x),
        y=mp.FixedMessage(y),
        z=mp.NormalMessage.from_mode(
            np.zeros(n_observations), 100
        ),
    )

    opt = mp.optimise.LaplaceOptimiser(
        model_approx,
        n_iter=3
    )
    opt.run()

    for string in ("centre", "intensity", "sigma"):
        print(f"{string} = {opt.model_approx[string].mu}")
