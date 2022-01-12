import configparser
import logging

import numpy as np

from autoconf import conf
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit import exc

logger = logging.getLogger(
    __name__
)


class Initializer:
    def __init__(
            self,
            lower_limit: float,
            upper_limit: float
    ):
        """
        The Initializer creates the initial set of samples in non-linear parameter space that can be passed into a
        `NonLinearSearch` to define where to begin sampling.

        Although most non-linear searches have in-built functionality to do this, some do not cope well with parameter
        resamples that are raised as FitException's. Thus, PyAutoFit uses its own initializer to bypass these problems.
        """
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    @classmethod
    def from_config(cls, config):
        """
        Load the Initializer from a non_linear config file.
        """

        try:

            initializer = config("initialize", "method")

        except configparser.NoSectionError:

            return None

        if initializer in "prior":

            return InitializerPrior()

        elif initializer in "ball":

            ball_lower_limit = config("initialize", "ball_lower_limit")
            ball_upper_limit = config("initialize", "ball_upper_limit")

            return InitializerBall(
                lower_limit=ball_lower_limit, upper_limit=ball_upper_limit
            )

    def samples_from_model(
            self,
            total_points: int,
            model: AbstractPriorModel,
            fitness_function,
            use_prior_medians:bool=False
    ):
        """
        Generate the initial points of the non-linear search, by randomly drawing unit values from a uniform
        distribution between the ball_lower_limit and ball_upper_limit values.

        Parameters
        ----------
        total_points
            The number of points in non-linear paramemter space which initial points are created for.
        model
            An object that represents possible instances of some model with a given dimensionality which is the number
            of free dimensions of the model.
        """

        if conf.instance["general"]["test"]["test_mode"]:
            return self.samples_in_test_mode(total_points=total_points, model=model)

        logger.info("Generating initial samples of model, which are subject to prior limits and other constraints.")

        unit_parameter_lists = []
        parameter_lists = []
        figures_of_merit_list = []

        point_index = 0

        while point_index < total_points:

            if not use_prior_medians:

                unit_parameter_list = model.random_unit_vector_within_limits(
                    lower_limit=self.lower_limit, upper_limit=self.upper_limit
                )

                try:
                    parameter_list = model.vector_from_unit_vector(unit_vector=unit_parameter_list)
                except exc.PriorLimitException:
                    continue

            else:

                unit_parameter_list = [0.5]*model.prior_count
                parameter_list = model.vector_from_unit_vector(unit_vector=unit_parameter_list)

            try:
                figure_of_merit = fitness_function.figure_of_merit_from(
                    parameter_list=parameter_list
                )

                if np.isnan(figure_of_merit):
                    raise exc.FitException

                unit_parameter_lists.append(unit_parameter_list)
                parameter_lists.append(parameter_list)
                figures_of_merit_list.append(figure_of_merit)
                point_index += 1
            except exc.FitException:
                pass

        return unit_parameter_lists, parameter_lists, figures_of_merit_list

    def samples_in_test_mode(
            self,
            total_points: int,
            model: AbstractPriorModel
    ):
        """
        Generate the initial points of the non-linear search in test mode. Like normal, test model draws points, by
        randomly drawing unit values from a uniform distribution between the ball_lower_limit and ball_upper_limit
        values.

        However, the log likelihood function is bypassed and all likelihoods are returned with a value -1.0e99. This
        is so that integration testing of large-scale model-fitting projects can be performed efficiently by bypassing
        sampling of points using the `log_likelihood_function`.

        Parameters
        ----------
        total_points
            The number of points in non-linear paramemter space which initial points are created for.
        model
            An object that represents possible instances of some model with a given dimensionality which is the number
            of free dimensions of the model.
        """

        unit_parameter_lists = []
        parameter_lists = []
        figure_of_merit_list = []

        point_index = 0

        while point_index < total_points:

            unit_parameter_list = model.random_unit_vector_within_limits(
                lower_limit=self.lower_limit, upper_limit=self.upper_limit
            )
            parameter_list = model.vector_from_unit_vector(unit_vector=unit_parameter_list)
            unit_parameter_lists.append(unit_parameter_list)
            parameter_lists.append(parameter_list)
            figure_of_merit_list.append(-1.0e99)
            point_index += 1

        return unit_parameter_lists, parameter_lists, figure_of_merit_list


class InitializerPrior(Initializer):
    def __init__(self):
        """
        The Initializer creates the initial set of samples in non-linear parameter space that can be passed into a
        `NonLinearSearch` to define where to begin sampling.

        Although most non-linear searches have in-built functionality to do this, some do not cope well with parameter
        resamples that are raised as FitException's. Thus, PyAutoFit uses its own initializer to bypass these problems.

        The InitializerPrior class generates from the priors, by drawing all values as unit values between 0.0 and 1.0
        and mapping them to physical values via the prior.
        """
        super().__init__(lower_limit=0.0, upper_limit=1.0)


class InitializerBall(Initializer):
    def __init__(
            self,
            lower_limit: float,
            upper_limit: float
    ):
        """
        The Initializer creates the initial set of samples in non-linear parameter space that can be passed into a
        `NonLinearSearch` to define where to begin sampling.

        Although most non-linear searches have in-built functionality to do this, some do not cope well with parameter
        resamples that are raised as FitException's. Thus, PyAutoFit uses its own initializer to bypass these problems.

        The InitializerBall class generates the samples in a small compact volume or 'ball' in parameter space, which is
        the recommended initialization strategy for the MCMC `NonLinearSearch` Emcee.

        Parameters
        ----------
        lower_limit
            The lower limit of the uniform distribution unit values are drawn from when initializing walkers in a small
            compact ball.
        upper_limit
            The upper limit of the uniform distribution unit values are drawn from when initializing walkers in a small
            compact ball.
        """
        super().__init__(lower_limit=lower_limit, upper_limit=upper_limit)
