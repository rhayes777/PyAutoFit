import jax
import logging
import numpy as np
import os
import time

from timeout_decorator import timeout
from typing import Optional

from autoconf import conf
from autoconf import cached_property

from autofit import jax_wrapper
from autofit.jax_wrapper import numpy as xp
from autofit import exc


from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.non_linear.analysis import Analysis


def get_timeout_seconds():

    try:
        return conf.instance["general"]["test"]["lh_timeout_seconds"]
    except KeyError:
        pass

logger = logging.getLogger(__name__)
timeout_seconds = get_timeout_seconds()

class Fitness:
    def __init__(
        self,
        model : AbstractPriorModel,
        analysis : Analysis,
        paths : Optional[AbstractPaths] = None,
        fom_is_log_likelihood: bool = True,
        resample_figure_of_merit: float = -xp.inf,
        convert_to_chi_squared: bool = False,
        store_history: bool = False,
        use_jax_vmap : bool = False
    ):
        """
        Interfaces with any non-linear search to fit the model to the data and return a log likelihood via
        the analysis.

        The interface of a non-linear search and fitness function is summarised as follows:

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

        Many searches do not store the history of the parameters and log likelihood values, often to save
        memory on large model-fits. However, this can be useful, for example to plot the results of a model-fit
        versus iteration number. If the `store_history` bool is `True`, the parameters and log likelihoods are stored
        in the `parameters_history_list` and `figure_of_merit_history_list` attribute of the fitness object.

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
        store_history
            If `True`, the parameters and log likelihood values of every model-fit are stored in lists.
        """

        self.analysis = analysis
        self.model = model
        self.paths = paths
        self.fom_is_log_likelihood = fom_is_log_likelihood
        self.resample_figure_of_merit = resample_figure_of_merit
        self.convert_to_chi_squared = convert_to_chi_squared
        self.store_history = store_history

        self.parameters_history_list = []
        self.log_likelihood_history_list = []

        self.use_jax_vmap = use_jax_vmap

        self._call = self.call

        if jax_wrapper.use_jax:
            if self.use_jax_vmap:
                self._call = self._vmap

        if self.paths is not None:
            self.check_log_likelihood(fitness=self)

    def call(self, parameters):
        """
        A private method that calls the fitness function with the given parameters and additional keyword arguments.
        This method is intended for internal use only.

        Parameters
        ----------
        parameters
            The parameters (typically a list) chosen by a non-linear search, which are mapped to an instance of the
            model via its priors and fitted to the data.
        kwargs
            Additional key-word arguments that may be necessary for specific non-linear searches.

        Returns
        -------
        The figure of merit returned to the non-linear search, which is either the log likelihood or log posterior.
        """

        # Get instance from model
        instance = self.model.instance_from_vector(vector=parameters)

        # Evaluate log likelihood (must be side-effect free and exception-free)
        log_likelihood = self.analysis.log_likelihood_function(instance=instance)

        # Penalize NaNs in the log-likelihood
        log_likelihood = xp.where(xp.isnan(log_likelihood), self.resample_figure_of_merit, log_likelihood)

        # Determine final figure of merit
        if self.fom_is_log_likelihood:
            figure_of_merit = log_likelihood
        else:
            # Ensure prior list is compatible with JAX (must return a JAX array, not list)
            log_prior_array = xp.array(self.model.log_prior_list_from_vector(vector=parameters))
            figure_of_merit = log_likelihood + xp.sum(log_prior_array)

        # Convert to chi-squared scale if requested
        if self.convert_to_chi_squared:
            figure_of_merit *= -2.0

        return figure_of_merit

    def call_wrap(self, parameters):
        """
        Wrapper around a JAX-jitted likelihood function that optionally stores
        the history of evaluated parameters and likelihood values.

        Depending on whether the figure of merit
        (FoM) is defined as a log-likelihood (`self.fom_is_log_likelihood`), it
        either uses the FoM directly or subtracts the summed log-prior to obtain
        the log-likelihood.

        If `self.store_history` is True, both the input parameters and the
        corresponding log-likelihood are appended to internal history lists
        (`self.parameters_history_list`, `self.log_likelihood_history_list`).

        Parameters
        ----------
        parameters
            A vector of model parameters to evaluate.

        Returns
        -------
        float
            The computed figure of merit for the input parameters. This is either
            the log-likelihood itself or another objective function value,
            depending on configuration.
        """
        if self.use_jax_vmap:
            if len(np.array(parameters).shape) == 1:
                parameters = np.array(parameters)[None, :]

        figure_of_merit = self._call(parameters)

        print(figure_of_merit)

        if self.fom_is_log_likelihood:
            log_likelihood = figure_of_merit
        else:
            log_prior_list = xp.array(self.model.log_prior_list_from_vector(vector=parameters))
            log_likelihood = figure_of_merit - xp.sum(log_prior_list)

        if self.store_history:

            self.parameters_history_list.append(parameters)
            self.log_likelihood_history_list.append(log_likelihood)

        return figure_of_merit

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
        return self.call_wrap(parameters)

    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     # Remove non-pickleable attributes
    #     state.pop('_call', None)
    #     state.pop('_grad', None)
    #     return state
    #
    # def __setstate__(self, state):
    #     self.__dict__.update(state)

    @cached_property
    def _vmap(self):
        """
        Vectorized and JIT-compiled likelihood function.

        This wraps the base likelihood function (`self.call`) with both
        `jax.jit` and `jax.vmap`, producing a function that can evaluate
        batches of parameter vectors efficiently in parallel. The first
        call incurs compilation time, but subsequent calls are highly
        optimized.

        Because this is a `cached_property`, the compiled function is stored
        after its first creation, avoiding repeated JIT compilation overhead.
        """
        start = time.time()
        logger.info("JAX: Applying vmap and jit to likelihood function -- may take a few seconds.")
        func = jax.vmap(jax.jit(self.call))
        logger.info(f"JAX: vmap and jit applied in {time.time() - start} seconds.")
        return func

    @cached_property
    def _jit(self):
        """
        JIT-compiled likelihood function.

        This wraps the base likelihood function (`self.call`) with `jax.jit`,
        producing a compiled version optimized for repeated evaluation on a
        single set of parameters. The first call triggers compilation, while
        later calls benefit from the compiled execution.

        As a `cached_property`, the compiled function is cached after its
        first use, so JIT compilation only occurs once.
        """
        start = time.time()
        logger.info("JAX: Applying jit to likelihood function -- may take a few seconds.")
        func = jax_wrapper.jit(self.call)
        logger.info(f"JAX: jit applied in {time.time() - start} seconds.")
        return func

    @cached_property
    def _grad(self):
        """
        Gradient of the JIT-compiled likelihood function.

        This wraps the JIT-compiled likelihood function (`self._call`) with
        `jax.grad`, returning a function that computes gradients of the
        likelihood with respect to its input parameters. Useful for gradient-
        based optimization and inference methods.

        Since this is a `cached_property`, the gradient function is compiled
        and cached on first access, ensuring that expensive setup is done
        only once.
        """
        start = time.time()
        logger.info("JAX: Applying grad to likelihood function -- may take a few seconds.")
        func = jax_wrapper.grad(self.call)
        logger.info(f"JAX: grad applied in {time.time() - start} seconds.")
        return func

    def grad(self, *args, **kwargs):
        return self._grad(*args, **kwargs)

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
        import numpy as np

        if os.environ.get("PYAUTOFIT_TEST_MODE") == "1":
            return

        if not conf.instance["general"]["test"]["check_likelihood_function"]:
            return

        try:
            samples_summary = self.paths.load_samples_summary()
        except FileNotFoundError:
            return

        try:
            max_log_likelihood_sample = samples_summary.max_log_likelihood_sample
        except AttributeError:
            return
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