from autofit import exc

from autofit.non_linear.log import logger

import configparser


class Initializer:
    def __init__(self, lower_limit, upper_limit):
        """
        The Initializer creates the initial set of samples in non-linear parameter space that can be passed into a
        non-linear search to define where to begin sampling.

        Although most non-linear searches have in-built functionality to do this, some do not cope well with parameter
        resamples that are raised as FitException's. Thus, PyAutoFit uses its own initializer to bypass these problems.

        Parameters
        ----------
        """
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    @classmethod
    def from_config(cls, config):
        """Load the Initializer from a non_linear config file."""

        try:

            initializer = config("initialize", "method", str)

        except configparser.NoSectionError:

            return None

        if initializer in "prior":

            return InitializerPrior()

        elif initializer in "ball":

            ball_lower_limit = config("initialize", "ball_lower_limit", float)
            ball_upper_limit = config("initialize", "ball_upper_limit", float)

            return InitializerBall(
                lower_limit=ball_lower_limit, upper_limit=ball_upper_limit
            )

    def initial_samples_from_model(self, total_points, model, fitness_function):
        """Generate the initial points of the non-linear search, by randomly drawing unit values from a uniform
        distribution between the ball_lower_limit and ball_upper_limit values.

        Parameters
        ----------
        total_points : int
            The number of points in non-linear paramemter space which initial points are created for.
        model : ModelMapper
            An object that represents possible instances of some model with a given dimensionality which is the number
            of free dimensions of the model.
        """

        logger.info("Generating initial samples of model, which are subject to prior limits and other constraints.")

        initial_unit_parameters = []
        initial_parameters = []
        initial_figures_of_merit = []

        point_index = 0

        while point_index < total_points:

            unit_parameters = model.random_unit_vector_within_limits(
                lower_limit=self.lower_limit, upper_limit=self.upper_limit
            )
            parameters = model.vector_from_unit_vector(unit_vector=unit_parameters)

            try:
                figure_of_merit = fitness_function.figure_of_merit_from_parameters(
                    parameters=parameters
                )
                initial_unit_parameters.append(unit_parameters)
                initial_parameters.append(parameters)
                initial_figures_of_merit.append(figure_of_merit)
                point_index += 1
            except exc.FitException:
                pass

        return initial_unit_parameters, initial_parameters, initial_figures_of_merit


class InitializerPrior(Initializer):
    def __init__(self):
        """
        The Initializer creates the initial set of samples in non-linear parameter space that can be passed into a
        non-linear search to define where to begin sampling.

        Although most non-linear searches have in-built functionality to do this, some do not cope well with parameter
        resamples that are raised as FitException's. Thus, PyAutoFit uses its own initializer to bypass these problems.

        The InitializerPrior class generates from the priors, by drawing all values as unit values between 0.0 and 1.0
        and mapping them to physical values via the prior.
        """

        super().__init__(lower_limit=0.0, upper_limit=1.0)


class InitializerBall(Initializer):
    def __init__(self, lower_limit, upper_limit):
        """
        The Initializer creates the initial set of samples in non-linear parameter space that can be passed into a
        non-linear search to define where to begin sampling.

        Although most non-linear searches have in-built functionality to do this, some do not cope well with parameter
        resamples that are raised as FitException's. Thus, PyAutoFit uses its own initializer to bypass these problems.

        The InitializerBall class generates the samples in a small compact volume or 'ball' in parameter space, which is
        the recommended initialization strategy for the MCMC non-linear search Emcee.

        Parameters
        ----------
        lower_limit : float
            The lower limit of the uniform distribution unit values are drawn from when initializing walkers in a small
            compact ball.
        upper_limit : float
            The upper limit of the uniform distribution unit values are drawn from when initializing walkers in a small
            compact ball.
        """

        super().__init__(lower_limit=lower_limit, upper_limit=upper_limit)
