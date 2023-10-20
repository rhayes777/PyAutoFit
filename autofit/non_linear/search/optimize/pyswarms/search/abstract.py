from os import path
from typing import Optional

import numpy as np

from autoconf import conf
from autofit import exc
from autofit.database.sqlalchemy_ import sa
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.fitness import Fitness
from autofit.non_linear.initializer import AbstractInitializer
from autofit.non_linear.search.optimize.abstract_optimize import AbstractOptimizer
from autofit.non_linear.samples.sample import Sample
from autofit.non_linear.samples.samples import Samples
from autofit.plot import PySwarmsPlotter
from autofit.plot.output import Output


class FitnessPySwarms(Fitness):

    def __call__(self, parameters, *kwargs):
        """
        Interfaces with any non-linear in order to fit a model to the data and return a log likelihood via
        an `Analysis` class.

        The interface is described in full in the `__init__` docstring.

        `PySwarms` have a unique interface in that lists of parameters corresponding to multiple particles are
        passed to the fitness function. A bespoke `__call__` method is therefore required to handle this.

        The figure of merit is the log posterior multiplied by -2.0, which is the chi-squared value that is minimized
        by the `PySwarms` non-linear search.

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

        if isinstance(parameters[0], float):
            parameters = [parameters]

        figure_of_merit_list = []

        for params_of_particle in parameters:

            try:
                instance = self.model.instance_from_vector(vector=params_of_particle)
                log_likelihood = self.analysis.log_likelihood_function(instance=instance)
                log_prior = self.model.log_prior_list_from_vector(vector=params_of_particle)
                log_posterior = log_likelihood + sum(log_prior)
                figure_of_merit = -2.0 * log_posterior
            except exc.FitException:
                figure_of_merit = np.nan

            if np.isnan(figure_of_merit):
                figure_of_merit = -2.0 * self.resample_figure_of_merit

            figure_of_merit_list.append(figure_of_merit)

        return np.asarray(figure_of_merit_list)



class AbstractPySwarms(AbstractOptimizer):
    def __init__(
            self,
            name: Optional[str] = None,
            path_prefix: Optional[str] = None,
            unique_tag: Optional[str] = None,
            initializer: Optional[AbstractInitializer] = None,
            iterations_per_update: int = None,
            number_of_cores: int = None,
            session: Optional[sa.orm.Session] = None,
            **kwargs
    ):
        """
        A PySwarms Particle Swarm Optimizer global non-linear search.

        For a full description of PySwarms, checkout its Github and readthedocs webpages:

        https://github.com/ljvmiranda921/pyswarms
        https://pyswarms.readthedocs.io/en/latest/index.html

        Parameters
        ----------
        name
            The name of the search, controlling the last folder results are output.
        path_prefix
            The path of folders prefixing the name folder where results are output.
        unique_tag
            The name of a unique tag for this model-fit, which will be given a unique entry in the sqlite database
            and also acts as the folder after the path prefix and before the search name.
        initializer
            Generates the initialize samples of non-linear parameter space (see autofit.non_linear.initializer).
        number_of_cores
            The number of cores sampling is performed using a Python multiprocessing Pool instance.
        session
            An SQLalchemy session instance so the results of the model-fit are written to an SQLite database.
        """

        number_of_cores = (
            self._config("parallel", "number_of_cores")
            if number_of_cores is None
            else number_of_cores
        )

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            initializer=initializer,
            iterations_per_update=iterations_per_update,
            number_of_cores=number_of_cores,
            session=session,
            **kwargs
        )

        self.logger.debug("Creating PySwarms Search")

    def _fit(self, model: AbstractPriorModel, analysis):
        """
        Fit a model using PySwarms and the Analysis class which contains the data and returns the log likelihood from
        instances of the model, which the `NonLinearSearch` seeks to maximize.

        Parameters
        ----------
        model : ModelMapper
            The model which generates instances for different points in parameter space.
        analysis : Analysis
            Contains the data and the log likelihood function which fits an instance of the model to the data, returning
            the log likelihood the `NonLinearSearch` maximizes.

        Returns
        -------
        A result object comprising the Samples object that inclues the maximum log likelihood instance and full
        chains used by the fit.
        """

        fitness = FitnessPySwarms(
            model=model,
            analysis=analysis,
            fom_is_log_likelihood=False,
            resample_figure_of_merit=-np.inf,
            convert_to_chi_squared=True
        )

        try:

            search_internal_dict = self.paths.load_search_internal()

            search_internal = search_internal_dict["pos_history"]

            init_pos = search_internal[-1]
            total_iterations = search_internal_dict["total_iterations"]

            self.logger.info(
                "Resuming PySwarms non-linear search (previous samples found)."
            )

        except (FileNotFoundError, TypeError):

            unit_parameter_lists, parameter_lists, log_posterior_list = self.initializer.samples_from_model(
                total_points=self.config_dict_search["n_particles"],
                model=model,
                fitness=fitness,
            )

            init_pos = np.zeros(shape=(self.config_dict_search["n_particles"], model.prior_count))

            for index, parameters in enumerate(parameter_lists):

                init_pos[index, :] = np.asarray(parameters)

            total_iterations = 0

            self.logger.info(
                "Starting new PySwarms non-linear search (no previous samples found)."
            )

        ## TODO : Use actual limits

        vector_lower = model.vector_from_unit_vector(
            unit_vector=[1e-6] * model.prior_count,
            ignore_prior_limits=True
        )
        vector_upper = model.vector_from_unit_vector(
            unit_vector=[0.9999999] * model.prior_count,
            ignore_prior_limits=True
        )

        lower_bounds = [lower for lower in vector_lower]
        upper_bounds = [upper for upper in vector_upper]

        bounds = (np.asarray(lower_bounds), np.asarray(upper_bounds))

        while total_iterations < self.config_dict_run["iters"]:

            search_internal = self.search_internal_from(
                model=model,
                fitness=fitness,
                bounds=bounds,
                init_pos=init_pos
            )

            iterations_remaining = self.config_dict_run["iters"] - total_iterations

            iterations = min(self.iterations_per_update, iterations_remaining)

            if iterations > 0:

                search_internal.optimize(objective_func=fitness.__call__, iters=iterations)

                total_iterations += iterations

                search_internal_dict = {
                    "pos_history" : search_internal.pos_history,
                    "total_iterations": total_iterations,
                    "log_posterior_list": [-0.5 * cost for cost in search_internal.cost_history],
                    "time": self.timer.time if self.timer else None,
                }

                self.paths.save_search_internal(
                    obj=search_internal_dict,
                )

                self.perform_update(
                    model=model,
                    analysis=analysis,
                    during_analysis=True,
                    search_internal=search_internal
                )

                init_pos = search_internal.pos_history[-1]

        return search_internal

    def samples_via_internal_from(self, model, search_internal=None):
        """
        Returns a `Samples` object from the pyswarms internal results.

        The samples contain all information on the parameter space sampling (e.g. the parameters,
        log likelihoods, etc.).

        The internal search results are converted from the native format used by the search to lists of values
        (e.g. `parameter_lists`, `log_likelihood_list`).

        Parameters
        ----------
        model
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        """

        if search_internal is not None:

            search_internal_dict = {
                "total_iterations": None,
                "log_posterior_list": [-0.5 * cost for cost in search_internal.cost_history],
                "time": self.timer.time if self.timer else None,
            }
            pos_history = search_internal.pos_history



        else:

            search_internal_dict = self.paths.load_search_internal()
            pos_history = search_internal_dict["pos_history"]

            search_internal_dict = {
                "total_iterations": search_internal_dict["total_iterations"],
                "log_posterior_list": search_internal_dict["log_posterior_list"],
                "time": search_internal_dict["time"]
            }

        parameter_lists = [
            param.tolist() for parameters in pos_history for param in parameters
        ]
        parameter_lists_2 = [parameters.tolist()[0] for parameters in pos_history]

        log_posterior_list = search_internal_dict["log_posterior_list"]
        log_prior_list = model.log_prior_list_from(parameter_lists=parameter_lists)
        log_likelihood_list = [lp - prior for lp, prior in zip(log_posterior_list, log_prior_list)]
        weight_list = len(log_likelihood_list) * [1.0]

        sample_list = Sample.from_lists(
            model=model,
            parameter_lists=parameter_lists_2,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list
        )

        return Samples(
            model=model,
            sample_list=sample_list,
            samples_info=search_internal_dict,
            search_internal=pos_history
        )

    def config_dict_with_test_mode_settings_from(self, config_dict):

        return {
            **config_dict,
            "iters": 1,
        }

    def search_internal_from(self, model, fitness, bounds, init_pos):
        raise NotImplementedError()

    def plot_results(self, samples):

        def should_plot(name):
            return conf.instance["visualize"]["plots_search"]["pyswarms"][name]

        plotter = PySwarmsPlotter(
            samples=samples,
            output=Output(path=self.paths.image_path / "search", format="png")
        )

        if should_plot("contour"):
            plotter.contour()

        if should_plot("cost_history"):
            plotter.cost_history()

        if should_plot("trajectories"):
            plotter.trajectories()

        if should_plot("time_series"):
            plotter.time_series()