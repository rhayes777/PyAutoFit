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

    likelihood = mp.Factor(
        _likelihood,
        z=z,
        y=y_
    )

    prior_model = mp.MeanFieldPriorModel(
        likelihood * gaussian,
        centre=prior,
        intensity=prior,
        sigma=prior
    )

    model_approx = prior_model.mean_field_approximation({
        x_: mp.FixedMessage(x),
        y_: mp.FixedMessage(y),
        z: mp.NormalMessage.from_mode(
            np.zeros(n_observations), 100
        ),
    })

    opt = mp.optimise.LaplaceOptimiser(
        model_approx,
        n_iter=3
    )

    opt.run()

    for variable in prior_model.prior_variables:
        print(f"{variable.name} = {opt.model_approx[variable].mu}")
