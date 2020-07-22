import numpy as np

import autofit as af
from autofit import message_passing as mp
from .model import Gaussian, make_data, _gaussian, _likelihood

n_observations = 100

prior = af.GaussianPrior(
    mean=0,
    sigma=40
)


def make_model(
        number,
        observations,
        intensity
):
    x_ = mp.Variable(
        f"x_{number}", observations
    )
    y_ = mp.Variable(
        f"y_{number}", observations
    )
    z = mp.Variable(
        f"z_{number}", observations
    )
    centre = mp.Variable(
        f"centre_{number}"
    )
    sigma = mp.Variable(
        f"sigma_{number}"
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

    prior_centre = mp.Factor(
        prior
    )(centre)
    prior_sigma = mp.Factor(
        prior
    )(sigma)

    return likelihood * gaussian * prior_centre * prior_sigma


def make_message_dict(
        number,
        gaussian
):
    x, y = make_data(
        gaussian,
        n_observations
    )

    return {
        f"centre_{number}": mp.NormalMessage.from_prior(
            prior
        ),
        f"intensity": mp.NormalMessage.from_prior(
            prior
        ),
        f"sigma_{number}": mp.NormalMessage.from_prior(
            prior
        ),
        f"x_{number}": mp.FixedMessage(x),
        f"y_{number}": mp.FixedMessage(y),
        f"z_{number}": mp.NormalMessage.from_mode(
            np.zeros(n_observations), 100
        )
    }


def test_gaussian():
    intensity = 25.0

    observations = mp.Plate(
        name="observations"
    )

    intensity_ = mp.Variable(
        "intensity"
    )

    prior_intensity = mp.Factor(
        prior
    )(intensity_)

    model = prior_intensity
    message_dict = dict()

    kwarg_list = [
        {"centre": 50.0, "sigma": 10.0},
        {"centre": 30.0, "sigma": 20.0},
        {"centre": 0.0, "sigma": 50.0},
    ]

    for number, kwargs in enumerate(kwarg_list):
        model *= make_model(
            number,
            observations,
            intensity_
        )

        message_dict.update(
            make_message_dict(
                number,
                Gaussian(
                    intensity=intensity,
                    **kwargs
                )
            )
        )

    model_approx = mp.MeanFieldApproximation.from_kws(
        model,
        **message_dict
    )

    opt = mp.optimise.LaplaceOptimiser(
        model_approx,
        n_iter=3
    )
    opt.run()

    def print_factor(name):
        print(f"{name} = {opt.model_approx[name].mu}")

    print_factor("intensity")

    for number in range(len(kwarg_list)):
        for string in (f"centre_{number}", f"sigma_{number}"):
            print_factor(string)
