import configparser
import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List

import numpy as np

from autofit import exc
from autofit.mapper.prior.abstract import Prior
from autofit.mapper.prior_model.abstract import AbstractPriorModel

logger = logging.getLogger(__name__)


class AbstractInitializer(ABC):
    """
    Family of classes used to provide initial points for non-linear search
    """

    @abstractmethod
    def _generate_unit_parameter_list(self, model):
        pass

    def samples_from_model(
        self,
        total_points: int,
        model: AbstractPriorModel,
        fitness,
        use_prior_medians: bool = False,
        test_mode_samples: bool = True,
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

        if os.environ.get("PYAUTOFIT_TEST_MODE") == "1" and test_mode_samples:
            return self.samples_in_test_mode(total_points=total_points, model=model)

        logger.info(
            "Generating initial samples of model, which are subject to prior limits and other constraints."
        )

        unit_parameter_lists = []
        parameter_lists = []
        figures_of_merit_list = []

        point_index = 0

        while point_index < total_points:
            if not use_prior_medians:
                unit_parameter_list = self._generate_unit_parameter_list(model)
            else:
                unit_parameter_list = [0.5] * model.prior_count

            parameter_list = model.vector_from_unit_vector(
                unit_vector=unit_parameter_list
            )

            try:
                figure_of_merit = fitness(parameters=parameter_list)

                if np.isnan(figure_of_merit) or figure_of_merit < -1e98:
                    raise exc.FitException

                unit_parameter_lists.append(unit_parameter_list)
                parameter_lists.append(parameter_list)
                figures_of_merit_list.append(figure_of_merit)
                point_index += 1
            except exc.FitException:
                pass


        if total_points > 1 and np.allclose(
            a=figures_of_merit_list[0], b=figures_of_merit_list[1:]
        ):
            raise exc.InitializerException(
                """
                The initial samples all have the same figure of merit (e.g. log likelihood values).
                
                The non-linear search will therefore not progress correctly.
                
                Possible causes for this behaviour are:
                
                - The `log_likelihood_function` of the analysis class is defined incorrectly.
                - The model parameterization creates numerically inaccurate log likelihoods.
                - The`log_likelihood_function`  is always returning `nan` values.            
                """
            )

        return unit_parameter_lists, parameter_lists, figures_of_merit_list

    def samples_in_test_mode(self, total_points: int, model: AbstractPriorModel):
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

        logger.warning(
            f"TEST MODE ON: SAMPLES BEING ASSIGNED ABRITRARY LARGE LIKELIHOODS"
        )

        unit_parameter_lists = []
        parameter_lists = []
        figure_of_merit_list = []

        point_index = 0

        figure_of_merit = -1.0e99

        while point_index < total_points:
            try:
                unit_parameter_list = self._generate_unit_parameter_list(model)
                parameter_list = model.vector_from_unit_vector(
                    unit_vector=unit_parameter_list
                )
                model.instance_from_vector(vector=parameter_list)
                unit_parameter_lists.append(unit_parameter_list)
                parameter_lists.append(parameter_list)
                figure_of_merit_list.append(figure_of_merit)
                figure_of_merit *= 10.0
                point_index += 1
            except exc.FitException:
                pass

        return unit_parameter_lists, parameter_lists, figure_of_merit_list


class SpecificRangeInitializer(AbstractInitializer):
    def __init__(
        self,
        parameter_dict: Dict[Prior, Tuple[float, float]],
        lower_limit=0.0,
        upper_limit=1.0,
    ):
        """
        Initializer that allows the range of possible starting points for each prior
        to be specified explicitly.

        Parameters
        ----------
        parameter_dict
            A dictionary mapping priors to inclusive ranges of physical values that
            the initial values for that dimension in the search may take
        lower_limit
            A default, unit lower limit used when a prior is not specified
        upper_limit
            A default, unit upper limit used when a prior is not specified
        """
        self.parameter_dict = parameter_dict
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def _generate_unit_parameter_list(self, model: AbstractPriorModel) -> List[float]:
        """
        Generate a unit vector for the model. The default limits are used for any
        priors which the model has but are not found in the parameter dict.

        Parameters
        ----------
        model
            A model for which initial points are required

        Returns
        -------
        A unit vector
        """

        unit_parameter_list = []
        for prior in model.priors_ordered_by_id:
            try:
                lower, upper = map(prior.unit_value_for, self.parameter_dict[prior])
                value = random.uniform(lower, upper)
            except KeyError:
                logger.warning(
                    f"Range for {'.'.join(model.path_for_prior(prior))} not set in the SpecificRangeInitializer. "
                    f"Using defaults."
                )
                lower = self.lower_limit
                upper = self.upper_limit

                value = prior.unit_value_for(prior.random(lower, upper))

            unit_parameter_list.append(value)

        return unit_parameter_list


class Initializer(AbstractInitializer):
    def __init__(self, lower_limit: float, upper_limit: float):
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

    def _generate_unit_parameter_list(self, model):
        return model.random_unit_vector_within_limits(
            lower_limit=self.lower_limit, upper_limit=self.upper_limit
        )


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
    def __init__(self, lower_limit: float, upper_limit: float):
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
