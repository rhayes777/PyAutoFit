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

    x_ = mp.Variable(
        "x", observations
    )
    y_ = mp.Variable(
        "y", observations
    )
    z = mp.Variable(
        "z", observations
    )

    gaussian = mp.Factor(
        _gaussian,
        x=x_
    ) == z

    centre = gaussian.centre
    intensity = gaussian.intensity
    sigma = gaussian.sigma

    likelihood = mp.Factor(
        _likelihood,
        z=z,
        y=y_
    )

    prior_centre = mp.Factor(
        prior,
        x=centre
    )
    prior_intensity = mp.Factor(
        prior,
        x=intensity
    )
    prior_sigma = mp.Factor(
        prior,
        x=sigma
    )

    model = likelihood * gaussian * prior_centre * prior_sigma * prior_intensity

    model_approx = mp.MeanFieldApproximation.from_kws(
        model,
        {
            centre: mp.NormalMessage.from_prior(
                prior
            ),
            intensity: mp.NormalMessage.from_prior(
                prior
            ),
            sigma: mp.NormalMessage.from_prior(
                prior
            ),
            x_: mp.FixedMessage(x),
            y_: mp.FixedMessage(y),
            z: mp.NormalMessage.from_mode(
                np.zeros(n_observations), 100
            ),
        }
    )

    opt = mp.optimise.LaplaceOptimiser(
        model_approx,
        n_iter=3
    )
    opt.run()

    for variable in (centre, intensity, sigma):
        print(f"{variable.name} = {opt.model_approx[variable].mu}")
