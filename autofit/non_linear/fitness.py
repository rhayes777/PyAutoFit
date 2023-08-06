import numpy as np

from autofit import exc

class Fitness:
    def __init__(
            self, model, analysis, fom_is_log_likelihood,
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
        """

        self.analysis = analysis
        self.model = model
        self.fom_is_log_likelihood = fom_is_log_likelihood

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

            if self.fom_is_log_likelihood:
                fom = self.log_likelihood_from(parameter_list=parameters)
            else:
                fom = self.log_posterior_from(parameter_list=parameters)

            figure_of_merit = self.figure_of_merit_from(parameter_list=parameters)

            if np.isnan(figure_of_merit):
                return self.resample_figure_of_merit

            return figure_of_merit

        except exc.FitException:
            return self.resample_figure_of_merit

    def fit_instance(self, instance):
        log_likelihood = self.analysis.log_likelihood_function(instance=instance)

        return log_likelihood

    def log_posterior_from(self, parameter_list):
        log_likelihood = self.log_likelihood_from(parameter_list=parameter_list)
        log_prior_list = self.model.log_prior_list_from_vector(
            vector=parameter_list
        )

        return log_likelihood + sum(log_prior_list)

    def figure_of_merit_from(self, parameter_list):
        """
        The figure of merit is the value that the `NonLinearSearch` uses to sample parameter space. This varies
        between different `NonLinearSearch`s, for example:

            - The *Optimizer* *PySwarms* uses the chi-squared value, which is the -2.0*log_posterior.
            - The *MCMC* algorithm *Emcee* uses the log posterior.
            - Nested samplers such as *Dynesty* use the log likelihood.
        """
        return

    @staticmethod
    def prior(cube, model):
        # NEVER EVER REFACTOR THIS LINE! Haha.

        phys_cube = model.vector_from_unit_vector(unit_vector=cube)

        for i in range(len(phys_cube)):
            cube[i] = phys_cube[i]

        return cube

    @staticmethod
    def fitness(cube, model, fitness):
        return fitness(instance=model.instance_from_vector(cube))

    @property
    def resample_figure_of_merit(self):
        """
        If a sample raises a FitException, this value is returned to signify that the point requires resampling or
         should be given a likelihood so low that it is discard.
        """
        return -np.inf