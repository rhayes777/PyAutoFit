from typing import Dict

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
        number: int,
        observations: mp.Plate,
        intensity: mp.Variable
) -> mp.FactorGraph:
    """
    Make a model that shares intensity but has its own variables for all other arguments.

    This model also includes its own fitness function, priors and deterministic variable.

    Parameters
    ----------
    number
        The number of the model
    observations
        A plate representing the observations dimension (1 for our 1D gaussian)
    intensity
        A shared variable representing the intensity of the gaussian

    Returns
    -------
    A part of the model for one of the gaussians
    """

    # x and y represent the data
    x_ = mp.Variable(
        f"x_{number}", observations
    )
    y_ = mp.Variable(
        f"y_{number}", observations
    )

    # z is a deterministic variable. It's because the fitness function ought to be a function of a function.
    # Instead, we can make it a function of z and set z as a deterministic variable for that function
    z = mp.Variable(
        f"z_{number}", observations
    )
    centre = mp.Variable(
        f"centre_{number}"
    )
    sigma = mp.Variable(
        f"sigma_{number}"
    )

    # Wraps the function that creates data from the gaussian so it can be a factor in the graph. I'd quite
    # like to implicitly generate variables as this point but right now the plate logic stops that from
    #  being tractable
    gaussian = mp.Factor(
        _gaussian
    )(
        x_,
        centre,
        intensity,
        sigma
    ) == z
    # Likelihood function is another variable. Note how it's a function of data and a deterministic variable.
    likelihood = mp.Factor(
        _likelihood
    )(z, y_)

    # Here I've made factors around a real autofit prior! That's exciting isn't it?
    prior_centre = mp.Factor(
        prior
    )(centre)
    prior_sigma = mp.Factor(
        prior
    )(sigma)

    # Make part of the model in the form of a factor graph
    return likelihood * gaussian * prior_centre * prior_sigma


def make_message_dict(
        number: int,
        gaussian: Gaussian
) -> Dict[str, mp.AbstractMessage]:
    """
    Create a dictionary of initial messages.

    Parameters
    ----------
    number
        The number of the model
    gaussian
        The true gaussian

    Returns
    -------
    A dictionary of initial messages
    """
    x, y = make_data(
        gaussian,
        n_observations
    )

    return {
        # Oh look you can make a message from a prior!?
        f"centre_{number}": mp.NormalMessage.from_prior(
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


# Commented out this test as it takes a while to run. It's really an integration test and I'll promote it to that
#  lofty status when I'm ready to 'release' some form of MP
def _test_gaussian():
    # There's one global intensity
    intensity = 25.0

    # This represents the one dimension of the gaussian
    observations = mp.Plate(
        name="observations"
    )

    # There one global intensity variable
    intensity_ = mp.Variable(
        "intensity"
    )

    # And one global intensity prior...
    prior_intensity = mp.Factor(
        prior
    )(intensity_)

    # ...that's our initial model
    model = prior_intensity

    # As such we only have one message associated with intensity
    message_dict = {
        f"intensity": mp.NormalMessage.from_prior(
            prior
        )
    }

    # Here I've defined some other arguments for our gaussians
    kwarg_list = [
        {"centre": 50.0, "sigma": 10.0},
        {"centre": 30.0, "sigma": 20.0},
        {"centre": 0.0, "sigma": 50.0},
    ]

    for number, kwargs in enumerate(kwarg_list):
        # Extend our factor graph with the model for one gaussian
        model *= make_model(
            number,
            observations,
            intensity_
        )

        # Update our initial messages with messages for one gaussian
        message_dict.update(
            make_message_dict(
                number,
                Gaussian(
                    intensity=intensity,
                    **kwargs
                )
            )
        )

    # Create a mean field approximation from the model. That's where we represent factors distributions and the model
    # as the product of those distributions
    model_approx = mp.MeanFieldApproximation.from_kws(
        model,
        **message_dict
    )

    # Run the optimiser. This needs updating so that execution terminates with a condition based on the evidence
    # rather than just after three iterations.
    opt = mp.optimise.LaplaceOptimiser(
        model_approx,
        n_iter=3
    )
    opt.run()

    # Finally we print some stuff to check how everything turned out. Still not convinced we're achieving a better
    # accuracy for intensity with the combined model though?
    def print_factor(name):
        print(f"{name} = {opt.model_approx[name].mu}")

    print_factor("intensity")

    for number in range(len(kwarg_list)):
        for string in (f"centre_{number}", f"sigma_{number}"):
            print_factor(string)
