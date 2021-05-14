from os import path
import numpy as np
from typing import Optional
from sqlalchemy.orm import Session

from autoconf import conf

from autofit import exc
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.log import logger
from autofit.non_linear.optimize.abstract_optimize import AbstractOptimizer
from autofit.non_linear.samples import OptimizerSamples, Sample

from autofit.plot import PySwarmsPlotter
from autofit.plot.mat_wrap.wrap.wrap_base import Output

class AbstractPySwarms(AbstractOptimizer):
    def __init__(
            self,
            name=None,
            path_prefix=None,
            unique_tag : Optional[str] = None,
            prior_passer=None,
            initializer=None,
            iterations_per_update : int = None,
            number_of_cores : int = None,
            session : Optional[Session] = None,
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
        number_of_cores : int
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

        logger.debug("Creating PySwarms Search")

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
            """The figure of merit is the value that the `NonLinearSearch` uses to sample parameter space. *PySwarms*
            uses the chi-squared value, which is the -2.0*log_posterior."""
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

            init_pos = self.load_points[-1]
            total_iterations = self.load_total_iterations

            logger.info("Existing PySwarms samples found, resuming non-linear search.")

        else:

            initial_unit_parameter_lists, initial_parameter_lists, initial_log_posterior_list = self.initializer.initial_samples_from_model(
                total_points=self.config_dict_search["n_particles"],
                model=model,
                fitness_function=fitness_function,
            )

            init_pos = np.zeros(shape=(self.config_dict_search["n_particles"], model.prior_count))

            for index, parameters in enumerate(initial_parameter_lists):

                init_pos[index, :] = np.asarray(parameters)

            total_iterations = 0

            logger.info("No PySwarms samples found, beginning new non-linear search. ")

        ## TODO : Use actual limits

        vector_lower = model.vector_from_unit_vector(unit_vector=[1e-6] * model.prior_count)
        vector_upper = model.vector_from_unit_vector(unit_vector=[0.9999999] * model.prior_count)

        lower_bounds = []
        upper_bounds = []

        for lower in vector_lower:
            lower_bounds.append(lower)
        for upper in vector_upper:
            upper_bounds.append(upper)

        bounds = (np.asarray(lower_bounds), np.asarray(upper_bounds))

        logger.info("Running PySwarmsGlobal Optimizer...")

        while total_iterations < self.config_dict_run["iters"]:

            pso = self.sampler_from(
                model=model,
                fitness_function=fitness_function,
                bounds=bounds,
                init_pos=init_pos
            )

            iterations_remaining = self.config_dict_run["iters"] - total_iterations

            if self.iterations_per_update > iterations_remaining:
                iterations = iterations_remaining
            else:
                iterations = self.iterations_per_update

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

                init_pos = self.load_points[-1]

        logger.info("PySwarmsGlobal complete")

    def fitness_function_from_model_and_analysis(self, model, analysis, log_likelihood_cap=None):

        return PySwarmsGlobal.Fitness(
            paths=self.paths,
            model=model,
            analysis=analysis,
            samples_from_model=self.samples_from,
            log_likelihood_cap=log_likelihood_cap
        )

    def sampler_from(self, model, fitness_function, bounds, init_pos):
        raise NotImplementedError()

    def samples_from(self, model):

        return PySwarmsSamples(
            model=model,
            points=self.load_points,
            log_posterior_list=self.load_log_posterior_list,
            total_iterations=self.load_total_iterations,
            time=self.timer.time
        )

    @property
    def load_points(self):
        return self.paths.load_object(
            "points"
        )

    @property
    def load_log_posterior_list(self):
        return self.paths.load_object(
            "log_posterior_list"
        )

    @property
    def load_total_iterations(self):
        return self.paths.load_object(
            "total_iterations"
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

class PySwarmsGlobal(AbstractPySwarms):

    __identifier_fields__ = (
        "n_particles",
        "cognitive",
        "social",
        "inertia",
    )

    def __init__(
            self,
            name=None,
            path_prefix=None,
            unique_tag : Optional[str] = None,
            prior_passer=None,
            initializer=None,
            iterations_per_update : int = None,
            number_of_cores : int = None,
            session : Optional[Session] = None,
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
        number_of_cores : int
            The number of cores Emcee sampling is performed using a Python multiprocessing Pool instance. If 1, a
            pool instance is not created and the job runs in serial.
        """

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            prior_passer=prior_passer,
            initializer=initializer,
            iterations_per_update=iterations_per_update,
            number_of_cores=number_of_cores,
            session=session,
            **kwargs
        )

        logger.debug("Creating PySwarms Search")

    def sampler_from(self, model, fitness_function, bounds, init_pos):
        """Get the static Dynesty sampler which performs the non-linear search, passing it all associated input Dynesty
        variables."""

        import pyswarms

        options = {
            "c1": self.config_dict_search["cognitive"],
            "c2": self.config_dict_search["social"],
            "w": self.config_dict_search["inertia"]
        }

        config_dict = self.config_dict_search
        config_dict.pop("cognitive")
        config_dict.pop("social")
        config_dict.pop("inertia")

        return pyswarms.global_best.GlobalBestPSO(
            dimensions=model.prior_count,
            bounds=bounds,
            init_pos=init_pos,
            options=options,
            **config_dict
        )


class PySwarmsLocal(AbstractPySwarms):

    __identifier_fields__ = (
        "n_particles",
        "cognitive",
        "social",
        "inertia",
        "number_of_k_neighbors",
        "minkowski_p_norm"
    )

    def __init__(
            self,
            name=None,
            path_prefix=None,
            unique_tag : Optional[str] = None,
            prior_passer=None,
            iterations_per_update : int = None,
            number_of_cores : int = None,
            session : Optional[Session] = None,
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
        number_of_cores : int
            The number of cores Emcee sampling is performed using a Python multiprocessing Pool instance. If 1, a
            pool instance is not created and the job runs in serial.
        """

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            prior_passer=prior_passer,
            iterations_per_update=iterations_per_update,
            number_of_cores=number_of_cores,
            session=session,
            **kwargs
        )

        logger.debug("Creating PySwarms Search")

    def sampler_from(self, model, fitness_function, bounds, init_pos):
        """
        Get the static Dynesty sampler which performs the non-linear search, passing it all associated input Dynesty
        variables.
        """

        import pyswarms

        options = {
            "c1": self.config_dict_search["cognitive"],
            "c2": self.config_dict_search["social"],
            "w": self.config_dict_search["inertia"],
            "k": self.config_dict_search["number_of_k_neighbors"],
            "p": self.config_dict_search["minkowski_p_norm"],
        }

        config_dict = self.config_dict_search
        config_dict.pop("cognitive")
        config_dict.pop("social")
        config_dict.pop("inertia")
        config_dict.pop("number_of_k_neighbors")
        config_dict.pop("minkowski_p_norm")

        return pyswarms.local_best.LocalBestPSO(
            dimensions=model.prior_count,
            bounds=bounds,
            init_pos=init_pos,
            options=options,
            **config_dict
        )


class PySwarmsSamples(OptimizerSamples):

    def __init__(
            self,
            model: AbstractPriorModel,
            points : np.ndarray,
            log_posterior_list : np.ndarray,
            total_iterations : int,
            time: Optional[float] = None,
    ):
        """
        Create an *OptimizerSamples* object from this non-linear search's output files on the hard-disk and model.

        For PySwarms, all quantities are extracted via pickled states of the particle and cost histories.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space. This maps the points from unit
            cube values to physical values via the priors.
        """

        self.points = points
        self._log_posterior_list = log_posterior_list
        self.total_iterations = total_iterations

        super().__init__(
            model=model,
            time=time,
        )

        self._samples = None

    @property
    def samples(self):
        """
        Create a `Samples` object from this non-linear search's output files on the hard-disk and model.

        For Emcee, all quantities are extracted via the hdf5 backend of results.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space. This maps the points from unit
            cube values to physical values via the priors.
        paths : af.Paths
            Manages all paths, e.g. where the search outputs are stored, the `NonLinearSearch` chains,
            etc.
        """

        if self._samples is not None:
            return self._samples

        parameter_lists = [
            param.tolist() for parameters in self.points for param in parameters
        ]
        log_prior_list = [
            sum(self.model.log_prior_list_from_vector(vector=vector)) for vector in parameter_lists
        ]
        log_likelihood_list = [lp - prior for lp, prior in zip(self._log_posterior_list, log_prior_list)]
        weight_list = len(log_likelihood_list) * [1.0]

        self._samples = Sample.from_lists(
            model=self.model,
            parameter_lists=[parameters.tolist()[0] for parameters in self.points],
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list
        )

        return self._samples