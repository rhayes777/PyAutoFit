import numpy as np
import os
from typing import Optional

from autoconf import conf

from autofit import exc

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.non_linear.analysis import Analysis

from timeout_decorator import timeout

from autofit import jax_wrapper


def get_timeout_seconds():

    try:
        return conf.instance["general"]["test"]["lh_timeout_seconds"]
    except KeyError:
        pass


timeout_seconds = get_timeout_seconds()


class Fitness:
    def __init__(
        self,
        model : AbstractPriorModel,
        analysis : Analysis,
        paths : Optional[AbstractPaths] = None,
        fom_is_log_likelihood: bool = True,
        resample_figure_of_merit: float = -np.inf,
        convert_to_chi_squared: bool = False,
    ):
        """
        Interfaces with any non-linear search to fit the model to the data and return a log likelihood via
        the analysis.

        The interface of a non-linear search and fitness function is summarized as follows:

        1) The non-linear search samples a new set of model parameters, which are passed to the fitness
        function's `__call__` method.

        2) The list of parameter values are mapped to an instance of the model.

        3) The instance is passed to the analysis class's log likelihood function, which fits the model to the
        data and returns the log likelihood.

        4) A final figure-of-merit is computed and returned to the non-linear search, which is either the log
        likelihood or log posterior (e.g. adding the log prior to the log likelihood).

        Certain searches (commonly nested samplers) require the parameters to be mapped from unit values to physical
        values, which is performed internally by the fitness object in step 2.

        Certain searches require the returned figure of merit to be a log posterior (often MCMC methods) whereas
        others require it to be a log likelihood (often nested samples which account for priors internally) in step 4.
        Which values is returned by the `fom_is_log_likelihood` bool.

        Some searches require a chi-squared value (which they minimized), given by the log likelihood multiplied
        by -2.0. This is returned by the fitness if the `convert_to_chi_squared` bool is `True`.

        If a model-fit raises an exception or returns a `np.nan`, a `resample_figure_of_merit` value is returned
        instead. The appropriate value depends on the search, but is typically either `None`, `-np.inf` or `1.0e99`.
        All values indicate to the non-linear search that the model-fit should be resampled or ignored.

        Parameters
        ----------
        analysis
            An object that encapsulates the data and a log likelihood function which fits the model to the data
            via the non-linear search.
        model
            The model that is fitted to the data, which is used by the non-linear search to create instances of
            the model that are fitted to the data via the log likelihood function.
        paths
            The paths of the search, which if the search is being resumed from an old run is used to check that
            the likelihood function has not changed from the previous run.
        fom_is_log_likelihood
            If `True`, the figure of merit returned by the fitness function is the log likelihood. If `False`, the
            figure of merit is the log posterior.
        resample_figure_of_merit
            The figure of merit returned if the model-fit raises an exception or returns a `np.nan`.
        convert_to_chi_squared
            If `True`, the figure of merit returned is the log likelihood multiplied by -2.0, such that it is a
            chi-squared value that is minimized.
        """

        self.analysis = analysis
        self.model = model
        self.paths = paths
        self.fom_is_log_likelihood = fom_is_log_likelihood
        self.resample_figure_of_merit = resample_figure_of_merit
        self.convert_to_chi_squared = convert_to_chi_squared
        self._log_likelihood_function = None

        if self.paths is not None:
            self.check_log_likelihood(fitness=self)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_log_likelihood_function"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._log_likelihood_function = None

    @property
    def log_likelihood_function(self):
        if self._log_likelihood_function is None:
            self._log_likelihood_function = jax_wrapper.jit(
                self.analysis.log_likelihood_function
            )

        return self._log_likelihood_function

    @timeout(timeout_seconds)
    def __call__(self, parameters, *kwargs):
        """
        Interfaces with any non-linear in order to fit a model to the data and return a log likelihood via
        an `Analysis` class.

        The interface is described in full in the `__init__` docstring above.

        Parameters
        ----------
        parameters
            The parameters (typically a list) chosen by a non-linear search, which are mapped to an instance of the
            model via its priors and fitted to the data.
        kwargs
            Addition key-word arguments that may be necessary for specific non-linear searches.

        Returns
        -------
        The figure of merit returned to the non-linear search, which is either the log likelihood or log posterior.
        """

        try:
            instance = self.model.instance_from_vector(vector=parameters)
            log_likelihood = self.log_likelihood_function(instance=instance)

            if np.isnan(log_likelihood):
                return self.resample_figure_of_merit

        except exc.FitException:
            return self.resample_figure_of_merit

        if self.fom_is_log_likelihood:
            figure_of_merit = log_likelihood
        else:
            log_prior_list = self.model.log_prior_list_from_vector(vector=parameters)
            figure_of_merit = log_likelihood + sum(log_prior_list)

        if self.convert_to_chi_squared:
            figure_of_merit *= -2.0

        return figure_of_merit

    def check_log_likelihood(self, fitness):
        """
        Changes to the PyAutoGalaxy source code may inadvertantly change the numerics of how a log likelihood is
        computed. Equally, one may set off a model-fit that resumes from previous results, but change the settings of
        the pixelization or inversion in a way that changes the log likelihood function.

        This function performs an optional sanity check, which raises an exception if the log likelihood calculation
        changes, to ensure a model-fit is not resumed with a different likelihood calculation to the previous run.

        If the model-fit has not been performed before (e.g. it is not a resume) this function outputs
        the `figure_of_merit` (e.g. the log likelihood) of the maximum log likelihood model at the end of the model-fit.

        If the model-fit is a resume, it loads this `figure_of_merit` and compares it against a new value computed for
        the resumed run (again using the maximum log likelihood model inferred). If the two likelihoods do not agree
        and therefore the log likelihood function has changed, an exception is raised and the code execution terminated.

        Parameters
        ----------
        paths
            certain searches the non-linear search outputs are stored,
            visualization, and pickled objects used by the database and aggregator.
        result
            The result containing the maximum log likelihood fit of the model.
        """

        if os.environ.get("PYAUTOFIT_TEST_MODE") == "1":
            return

        if not conf.instance["general"]["test"]["check_likelihood_function"]:
            return

        try:
            samples_summary = self.paths.load_samples_summary()
        except FileNotFoundError:
            return

        max_log_likelihood_sample = samples_summary.max_log_likelihood_sample
        log_likelihood_old = samples_summary.max_log_likelihood_sample.log_likelihood

        parameters = max_log_likelihood_sample.parameter_lists_for_model(model=self.model)

        log_likelihood_new = fitness(parameters=parameters)

        if not np.isclose(log_likelihood_old, log_likelihood_new):
            raise exc.SearchException(
                f"""
                Figure of merit sanity check failed. 

                This means that the existing results of a model fit used a different
                likelihood function compared to the one implemented now.
                Old Figure of Merit = {log_likelihood_old}
                New Figure of Merit = {log_likelihood_new}
                """
            )


