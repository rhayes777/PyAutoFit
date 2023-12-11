import numpy as np

from autoconf import conf

from autofit import exc

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
        model,
        analysis,
        fom_is_log_likelihood: bool = True,
        resample_figure_of_merit: float = -np.inf,
        convert_to_chi_squared: bool = False,
    ):
        """
        Interfaces with any non-linear in order to fit a model to the data and return a log likelihood via
        an `Analysis` class.

        The interface of a non-linear search and a fitness function can be summarized as follows:

        1) The non-linear search chooses a new set of parameters for the model, which are passed to the fitness
        function's `__call__` method.

        2) The parameter values (typically a list) are mapped to an instance of the model (via its priors if
        appropriate for the non-linear search).

        3) The instance is passed to the analysis class's log likelihood function, which fits the model to the
        data and returns the log likelihood.

        4) A final figure-of-merit is computed and returned to the non-linear search, which is either the log
        likelihood or log posterior depending on the type of non-linear search.

        It is common for nested sampling algorithms to require that the figure of merit returned is a log likelihood
        as priors are often built into the mapping of values from a unit hyper-cube. Optimizers and MCMC methods
        typically require that the figure of merit returned is a log posterior, with the prior terms added via this
        fitness function. This is not a strict rule, but is a good default.

        Some methods also require a chi-squared value to be computed (which is minimized), which is the log likelihood
        multiplied by -2.0. The `Fitness` class can also compute this value, if the `convert_to_chi_squared` bool is
        `True`.

        If a model-fit raises an exception of returns a `np.nan` a `resample_figure_of_merit` value is returned. The
        appropriate value depends on the non-linear search, but is typically either `None`, `-np.inf` or `1.0e99`.
        All values indicate to the non-linear search that the model-fit should be resampled or ignored.

        Parameters
        ----------
        analysis
            An object that encapsulates the data and a log likelihood function which fits the model to the data
            via the non-linear search.
        model
            The model that is fitted to the data, which is used by the non-linear search to create instances of
            the model that are fitted to the data via the log likelihood function.
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
        self.fom_is_log_likelihood = fom_is_log_likelihood
        self.resample_figure_of_merit = resample_figure_of_merit
        self.convert_to_chi_squared = convert_to_chi_squared
        self._log_likelihood_function = None

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
