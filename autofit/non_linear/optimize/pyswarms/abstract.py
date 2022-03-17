from os import path
from typing import Optional

import numpy as np

from autoconf import conf
from autofit import exc
from autofit.database.sqlalchemy_ import sa
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.abstract_search import PriorPasser
from autofit.non_linear.initializer import Initializer
from autofit.non_linear.optimize.abstract_optimize import AbstractOptimizer
from autofit.non_linear.optimize.pyswarms.samples import PySwarmsSamples
from autofit.plot import PySwarmsPlotter
from autofit.plot.output import Output


class AbstractPySwarms(AbstractOptimizer):
    def __init__(
            self,
            name: Optional[str] = None,
            path_prefix: Optional[str] = None,
            unique_tag: Optional[str] = None,
            prior_passer: Optional[PriorPasser] = None,
            initializer: Optional[Initializer] = None,
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
        prior_passer
            Controls how priors are passed from the results of this `NonLinearSearch` to a subsequent non-linear search.
        initializer
            Generates the initialize samples of non-linear parameter space (see autofit.non_linear.initializer).
        number_of_cores
            The number of cores Emcee sampling is performed using a Python multiprocessing Pool instance. If 1, a
            pool instance is not created and the job runs in serial.
        session
            An SQLalchemy session instance so the results of the model-fit are written to an SQLite database.
        """

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            prior_passer=prior_passer,
            initializer=initializer,
            iterations_per_update=iterations_per_update,
            session=session,
            **kwargs
        )

        self.number_of_cores = (
            self._config("parallel", "number_of_cores")
            if number_of_cores is None
            else number_of_cores
        )

        self.logger.debug("Creating PySwarms Search")

    class Fitness(AbstractOptimizer.Fitness):
        def __call__(self, parameters):

            figures_of_merit = []

            for params_of_particle in parameters:

                try:
                    figure_of_merit = self.figure_of_merit_from(
                        parameter_list=params_of_particle
                    )
                except exc.FitException:
                    figure_of_merit = -2.0 * self.resample_figure_of_merit

                figures_of_merit.append(figure_of_merit)

            return np.asarray(figures_of_merit)

        def figure_of_merit_from(self, parameter_list):
            """
            The figure of merit is the value that the `NonLinearSearch` uses to sample parameter space.

            PySwarms uses the chi-squared value, which is the -2.0*log_posterior.
            """
            return -2.0 * self.log_posterior_from(parameter_list=parameter_list)

    def _fit(self, model: AbstractPriorModel, analysis, log_likelihood_cap=None):
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

        fitness_function = self.fitness_function_from_model_and_analysis(
            model=model, analysis=analysis
        )

        if self.paths.is_object("points"):

            init_pos = self.paths.load_object("points")[-1]
            total_iterations = self.paths.load_object("total_iterations")

            self.logger.info("Existing PySwarms samples found, resuming non-linear search.")

        else:

            unit_parameter_lists, parameter_lists, log_posterior_list = self.initializer.samples_from_model(
                total_points=self.config_dict_search["n_particles"],
                model=model,
                fitness_function=fitness_function,
            )

            init_pos = np.zeros(shape=(self.config_dict_search["n_particles"], model.prior_count))

            for index, parameters in enumerate(parameter_lists):

                init_pos[index, :] = np.asarray(parameters)

            total_iterations = 0

            self.logger.info("No PySwarms samples found, beginning new non-linear search. ")

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

        self.logger.info("Running PySwarmsGlobal Optimizer...")

        while total_iterations < self.config_dict_run["iters"]:

            pso = self.sampler_from(
                model=model,
                fitness_function=fitness_function,
                bounds=bounds,
                init_pos=init_pos
            )

            iterations_remaining = self.config_dict_run["iters"] - total_iterations

            iterations = min(self.iterations_per_update, iterations_remaining)

            if iterations > 0:

                pso.optimize(objective_func=fitness_function.__call__, iters=iterations)

                total_iterations += iterations

                self.paths.save_object(
                    "total_iterations",
                    total_iterations
                )
                self.paths.save_object(
                    "points",
                    pso.pos_history
                )
                self.paths.save_object(
                    "log_posterior_list",
                    [-0.5 * cost for cost in pso.cost_history]
                )

                self.perform_update(
                    model=model, analysis=analysis, during_analysis=True
                )

                init_pos = self.paths.load_object("points")[-1]

        self.logger.info("PySwarmsGlobal complete")

    def fitness_function_from_model_and_analysis(self, model, analysis, log_likelihood_cap=None):

        return AbstractPySwarms.Fitness(
            paths=self.paths,
            model=model,
            analysis=analysis,
            samples_from_model=self.samples_from,
            log_likelihood_cap=log_likelihood_cap
        )

    def sampler_from(self, model, fitness_function, bounds, init_pos):
        raise NotImplementedError()

    def samples_from(self, model):

        return PySwarmsSamples.from_results_internal(
            results_internal=self.paths.load_object("points"),
            model=model,
            log_posterior_list=self.paths.load_object("log_posterior_list"),
            total_iterations=self.paths.load_object("total_iterations"),
            time=self.timer.time
        )

    def plot_results(self, samples):

        def should_plot(name):
            return conf.instance["visualize"]["plots_search"]["pyswarms"][name]

        plotter = PySwarmsPlotter(
            samples=samples,
            output=Output(path=path.join(self.paths.image_path, "search"), format="png")
        )

        if should_plot("contour"):
            plotter.contour()

        if should_plot("cost_history"):
            plotter.cost_history()

        if should_plot("trajectories"):
            plotter.trajectories()

        if should_plot("time_series"):
            plotter.time_series()
