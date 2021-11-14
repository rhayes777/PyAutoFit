from os import path
from typing import List, Optional

import numpy as np
from sqlalchemy.orm import Session

from autoconf import conf
from autofit import exc
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.optimize.abstract_optimize import AbstractOptimizer
from autofit.non_linear.samples import OptimizerSamples, Sample
from autofit.non_linear.abstract_search import PriorPasser
from autofit.non_linear.initializer import Initializer
from autofit.plot import DrawerPlotter
from autofit.plot.mat_wrap.wrap.wrap_base import Output


class Drawer(AbstractOptimizer):

    __identifier_fields__ = (
        "n_particles",
        "cognitive",
        "social",
        "inertia",
    )

    def __init__(
            self,
            name: Optional[str] = None,
            path_prefix: Optional[str] = None,
            unique_tag: Optional[str] = None,
            prior_passer: Optional[PriorPasser] = None,
            initializer: Optional[Initializer] = None,
            iterations_per_update: int = None,
            number_of_cores: int = None,
            session: Optional[Session] = None,
            **kwargs
    ):
        """
        A Drawer Particle Swarm Optimizer global non-linear search.

        For a full description of Drawer, checkout its Github and readthedocs webpages:

        https://github.com/ljvmiranda921/drawer
        https://drawer.readthedocs.io/en/latest/index.html

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

        self.logger.debug("Creating Drawer Search")

    class Fitness(AbstractOptimizer.Fitness):

        def figure_of_merit_from(self, parameter_list):
            """
            The figure of merit is the value that the `NonLinearSearch` uses to sample parameter space.

            The `Drawer` search can use either the log posterior values or log likelihood values.
            """
            return -2.0 * self.log_posterior_from(parameter_list=parameter_list)

    def _fit(self, model: AbstractPriorModel, analysis, log_likelihood_cap=None):
        """
        Fit a model using Drawer and the Analysis class which contains the data and returns the log likelihood from
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

        total_draws = self.config_dict_search["total_draws"]

        self.logger.info(f"Performing DrawerSearch for a total of {total_draws} points.")

        unit_parameter_lists, parameter_lists, log_posterior_list = self.initializer.samples_from_model(
            total_points=self.config_dict_search["total_draws"],
            model=model,
            fitness_function=fitness_function,
        )

        self.paths.save_object(
            "parameter_lists",
            parameter_lists
        )
        self.paths.save_object(
            "log_posterior_list",
            log_posterior_list
        )

        self.perform_update(
            model=model, analysis=analysis, during_analysis=False
        )

        self.logger.info("Drawer complete")

    def fitness_function_from_model_and_analysis(self, model, analysis, log_likelihood_cap=None):

        return Drawer.Fitness(
            paths=self.paths,
            model=model,
            analysis=analysis,
            samples_from_model=self.samples_from,
            log_likelihood_cap=log_likelihood_cap,
        )

    def samples_from(self, model):

        parameter_lists = self.paths.load_object("parameter_lists")

        return DrawerSamples(
            model=model,
            parameter_lists=parameter_lists,
            log_posterior_list=self.paths.load_object("log_posterior_list"),
            time=self.timer.time
        )

    def plot_results(self, samples):

        def should_plot(name):
            return conf.instance["visualize"]["plots_search"]["drawer"][name]

        plotter = DrawerPlotter(
            samples=samples,
            output=Output(path=path.join(self.paths.image_path, "search"), format="png")
        )


class DrawerSamples(OptimizerSamples):

    def __init__(
            self,
            model: AbstractPriorModel,
            parameter_lists: List[List[float]],
            log_posterior_list: List[float],
            time: Optional[float] = None,
    ):
        """
        Create an *OptimizerSamples* object from this non-linear search's output files on the hard-disk and model.

        For Drawer, all quantities are extracted via pickled states of the particle and cost histories.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space. This maps the points from unit
            cube values to physical values via the priors.
        """

        self._log_posterior_list = log_posterior_list

        log_prior_list = [
            sum(model.log_prior_list_from_vector(vector=vector)) for vector in parameter_lists
        ]
        log_likelihood_list = [lp - prior for lp, prior in zip(self._log_posterior_list, log_prior_list)]
        weight_list = len(log_likelihood_list) * [1.0]

        sample_list = Sample.from_lists(
            model=model,
            parameter_lists=parameter_lists,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list
        )

        super().__init__(
            model=model,
            sample_list=sample_list,
            time=time,
        )
