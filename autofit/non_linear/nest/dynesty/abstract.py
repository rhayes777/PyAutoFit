from abc import ABC
from os import path
from typing import Optional

import numpy as np

from autoconf import conf
from autofit.database.sqlalchemy_ import sa
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.abstract_search import PriorPasser
from autofit.non_linear.nest.abstract_nest import AbstractNest
from autofit.non_linear.nest.dynesty.plotter import DynestyPlotter
from autofit.non_linear.nest.dynesty.samples import DynestySamples
from autofit.plot.output import Output


def prior_transform(cube, model):
    phys_cube = model.vector_from_unit_vector(
        unit_vector=cube,
        ignore_prior_limits=True
    )

    for i in range(len(phys_cube)):
        cube[i] = phys_cube[i]

    return cube


class AbstractDynesty(AbstractNest, ABC):

    def __init__(
            self,
            name: str = "",
            path_prefix: str = "",
            unique_tag: Optional[str] = None,
            prior_passer: PriorPasser = None,
            iterations_per_update: int = None,
            number_of_cores: int = None,
            session: Optional[sa.orm.Session] = None,
            **kwargs
    ):
        """
        A Dynesty non-linear search.

        For a full description of Dynesty, checkout its GitHub and readthedocs webpages:

        https://github.com/joshspeagle/dynesty
        https://dynesty.readthedocs.io/en/latest/index.html

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
        iterations_per_update
            The number of iterations performed between every Dynesty back-up (via dumping the Dynesty instance as a
            pickle).
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
            iterations_per_update=iterations_per_update,
            session=session,
            **kwargs
        )

        self.number_of_cores = (
            self._config("parallel", "number_of_cores")
            if number_of_cores is None
            else number_of_cores
        )

        self.logger.debug("Creating DynestyStatic Search")

    class Fitness(AbstractNest.Fitness):
        @property
        def resample_figure_of_merit(self):
            """
            If a sample raises a FitException, this value is returned to signify that the point requires resampling or
             should be given a likelihood so low that it is discard.

             -np.inf is an invalid sample value for Dynesty, so we instead use a large negative number."""
            return -1.0e99

    def _fit(self, model: AbstractPriorModel, analysis, log_likelihood_cap=None):
        """
        Fit a model using Dynesty and the Analysis class which contains the data and returns the log likelihood from
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
        A result object comprising the Samples object that includes the maximum log likelihood instance and full
        set of accepted samples of the fit.
        """

        fitness_function = self.fitness_function_from_model_and_analysis(
            model=model, analysis=analysis, log_likelihood_cap=log_likelihood_cap,
        )

        if self.paths.is_object("dynesty"):

            sampler = self.paths.load_object(
                "dynesty"
            )
            sampler.loglikelihood = fitness_function

            sampler.rstate = np.random

            if self.number_of_cores == 1:
                sampler.M = map
            else:
                pool = self.make_sneaky_pool(
                    fitness_function
                )

                sampler.M = pool.map
                sampler.pool = pool

            self.logger.info("Existing Dynesty samples found, resuming non-linear search.")

        else:
            if self.number_of_cores >= 1:
                pool = self.make_sneaky_pool(
                    fitness_function
                )
            else:
                pool = None

            sampler = self.sampler_from(
                model=model,
                fitness_function=fitness_function,
                pool=pool
            )

            self.logger.info("No Dynesty samples found, beginning new non-linear search. ")

        if not hasattr(sampler, "pool"):
            sampler.pool = None

        finished = False

        while not finished:

            try:
                total_iterations = np.sum(sampler.results.ncall)
            except AttributeError:
                total_iterations = 0

            if self.config_dict_run["maxcall"] is not None:
                iterations = self.config_dict_run["maxcall"] - total_iterations
            else:
                iterations = self.iterations_per_update

            if iterations > 0:

                for i in range(10):

                    try:

                        config_dict_run = self.config_dict_run
                        config_dict_run.pop("maxcall")

                        sampler.run_nested(
                            maxcall=iterations,
                            print_progress=not self.silence,
                            **config_dict_run
                        )

                        if i == 9:
                            raise ValueError("Dynesty crashed due to repeated bounding errors")

                        break

                    except (ValueError, np.linalg.LinAlgError):

                        continue

            sampler.loglikelihood = None

            self.paths.save_object(
                "dynesty",
                sampler
            )

            sampler.loglikelihood = fitness_function

            self.perform_update(model=model, analysis=analysis, during_analysis=True)

            iterations_after_run = np.sum(sampler.results.ncall)

            if (
                    total_iterations == iterations_after_run
                    or total_iterations == self.config_dict_run["maxcall"]
            ):
                finished = True

    def sampler_from(
            self,
            model,
            fitness_function,
            pool=None
    ):
        return NotImplementedError()

    def samples_from(self, model):
        """
        Create a `Samples` object from this non-linear search's output files on the hard-disk and model.

        For Dynesty, all information that we need is available from the instance of the dynesty sampler.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space. This maps the points from unit
            cube values to physical values via the priors.
        """
        sampler = self.paths.load_object(
            "dynesty"
        )

        return DynestySamples.from_results_internal(
            model=model,
            results_internal=sampler.results,
            number_live_points=self.total_live_points,
            unconverged_sample_size=1,
            time=self.timer.time,
        )

    def samples_via_results_from(self, model):
        """
        Create a `Samples` object from this non-linear search's output files on the hard-disk and model.

        For Dynesty, all information that we need is available from the instance of the dynesty sampler.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space. This maps the points from unit
            cube values to physical values via the priors.
        """
        results_internal = self.paths.load_object(
            "results"
        )

        return DynestySamples.from_results_internal(
            model=model,
            results_internal=results_internal,
            number_live_points=self.total_live_points,
            unconverged_sample_size=1,
            time=self.timer.time,
        )

    def live_points_from_model_and_fitness_function(
            self, model, fitness_function
    ):

        unit_parameters, parameters, log_likelihood_list = self.initializer.samples_from_model(
            total_points=self.total_live_points,
            model=model,
            fitness_function=fitness_function,
        )

        init_unit_parameters = np.zeros(shape=(self.total_live_points, model.prior_count))
        init_parameters = np.zeros(shape=(self.total_live_points, model.prior_count))
        init_log_likelihood_list = np.zeros(shape=(self.total_live_points))

        for index in range(len(parameters)):
            init_unit_parameters[index, :] = np.asarray(unit_parameters[index])
            init_parameters[index, :] = np.asarray(parameters[index])
            init_log_likelihood_list[index] = np.asarray(log_likelihood_list[index])

        return [init_unit_parameters, init_parameters, init_log_likelihood_list]

    def remove_state_files(self):
        self.paths.remove_object("dynesty")

    @property
    def total_live_points(self):
        raise NotImplementedError()

    def plot_results(self, samples):

        if not samples.pdf_converged:
            return

        def should_plot(name):
            return conf.instance["visualize"]["plots_search"]["dynesty"][name]

        plotter = DynestyPlotter(
            samples=samples,
            output=Output(path=path.join(self.paths.image_path, "search"), format="png")
        )

        if should_plot("cornerplot"):
            plotter.cornerplot()

        if should_plot("runplot"):
            plotter.runplot()

        if should_plot("traceplot"):
            plotter.traceplot()

        if should_plot("cornerpoints"):
            plotter.cornerpoints()
