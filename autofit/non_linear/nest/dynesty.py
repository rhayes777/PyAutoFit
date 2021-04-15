import sys
from abc import ABC

import numpy as np
from dynesty import NestedSampler as StaticSampler
from dynesty.dynesty import DynamicNestedSampler

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.log import logger
from autofit.non_linear.abstract_search import PriorPasser
from autofit.non_linear.nest.abstract_nest import AbstractNest
from autofit.non_linear.result import Result
from autofit.non_linear.samples import NestSamples, Sample


def prior_transform(cube, model):

    phys_cube = model.vector_from_unit_vector(unit_vector=cube)

    for i in range(len(phys_cube)):
        cube[i] = phys_cube[i]

    return cube


class AbstractDynesty(AbstractNest, ABC):
    def __init__(
            self,
            name : str ="",
            path_prefix : str= "",
            prior_passer : PriorPasser = None,
            iterations_per_update=None,
            number_of_cores=None,
            session=None,
            **kwargs
    ):
        """
        A Dynesty non-linear search.

        For a full description of Dynesty, checkout its GitHub and readthedocs webpages:

        https://github.com/joshspeagle/dynesty
        https://dynesty.readthedocs.io/en/latest/index.html

        Parameters
        ----------
        name : str
            The name of the search, controlling the last folder results are output.
        path_prefix : str
            The path of folders prefixing the name folder where results are output.
        prior_passer : af.PriorPasser
            Controls how priors are passed from the results of this `NonLinearSearch` to a subsequent non-linear search.
        iterations_per_update : int
            The number of iterations performed between every Dynesty back-up (via dumping the Dynesty instance as a
            pickle).
        number_of_cores : int
            The number of cores Emcee sampling is performed using a Python multiprocessing Pool instance. If 1, a
            pool instance is not created and the job runs in serial.
        """

        super().__init__(
            name=name,
            path_prefix=path_prefix,
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

        logger.debug("Creating DynestyStatic NLO")

    @property
    def no_limit(self):
        if self.config_dict["maxcall"] is None:
            return True
        return False

    def config_dict_run_nested_from(self, sampler):

        config_dict = self.config_dict
        config_dict_run_nested = {}

        for key in sampler.run_nested.__code__.co_varnames:
            try:
                config_dict_run_nested[key] = config_dict[key]
            except KeyError:
                pass

        config_dict_run_nested.pop("maxcall")

        return config_dict_run_nested

    class Fitness(AbstractNest.Fitness):
        @property
        def resample_figure_of_merit(self):
            """If a sample raises a FitException, this value is returned to signify that the point requires resampling or
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
        set of accepted ssamples of the fit.
        """

        pool, pool_ids = self.make_pool()

        fitness_function = self.fitness_function_from_model_and_analysis(
            model=model, analysis=analysis, pool_ids=pool_ids, log_likelihood_cap=log_likelihood_cap,
        )

        if self.paths.is_object("dynesty"):

            sampler = self.paths.load_object(
                "dynesty"
            )
            sampler.loglikelihood = fitness_function
            sampler.pool = pool

            sampler.rstate = np.random

            if self.number_of_cores == 1:
                sampler.M = map
            else:
                sampler.M = pool.map

            logger.info("Existing Dynesty samples found, resuming non-linear search.")

        else:

            sampler = self.sampler_from(
                model=model, fitness_function=fitness_function, pool=pool
            )

            logger.info("No Dynesty samples found, beginning new non-linear search. ")

        config_dict_run_nested = self.config_dict_run_nested_from(sampler=sampler)

        finished = False

        while not finished:

            try:
                total_iterations = np.sum(sampler.results.ncall)
            except AttributeError:
                total_iterations = 0

            if not self.no_limit:
                iterations = self.config_dict["maxcall"] - total_iterations
            else:
                iterations = self.iterations_per_update

            if iterations > 0:

                for i in range(10):

                    try:

                        sampler.run_nested(
                            maxcall=iterations,
                            print_progress=not self.silence,
                            **config_dict_run_nested
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
                    or total_iterations == self.config_dict["maxcall"]
            ):
                finished = True

    def sampler_from(self, model, fitness_function, pool):
        return NotImplementedError()

    def samples_via_sampler_from_model(self, model):
        """Create a `Samples` object from this non-linear search's output files on the hard-disk and model.

        For Dynesty, all information that we need is available from the instance of the dynesty sampler.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space. This maps the points from unit
            cube values to physical values via the priors.
        paths : af.Paths
            Manages all paths, e.g. where the search outputs are stored, the samples, etc.
        """
        sampler = self.paths.load_object(
            "dynesty"
        )
        parameters = sampler.results.samples.tolist()
        log_priors = [
            sum(model.log_priors_from_vector(vector=vector)) for vector in parameters
        ]
        log_likelihoods = list(sampler.results.logl)

        try:
            weights = list(
                np.exp(np.asarray(sampler.results.logwt) - sampler.results.logz[-1])
            )
        except:
            weights = sampler.results["weights"]

        total_samples = int(np.sum(sampler.results.ncall))
        log_evidence = np.max(sampler.results.logz)

        return NestSamples(
            model=model,
            samples=Sample.from_lists(
                log_priors=log_priors,
                log_likelihoods=log_likelihoods,
                weights=weights,
                model=model,
                parameters=parameters
            ),
            total_samples=total_samples,
            log_evidence=log_evidence,
            number_live_points=self.config_dict["nlive"],
            time=self.timer.time
        )

    def initial_live_points_from_model_and_fitness_function(
            self, model, fitness_function
    ):

        unit_parameters, parameters, log_likelihoods = self.initializer.initial_samples_from_model(
            total_points=self.config_dict["nlive"],
            model=model,
            fitness_function=fitness_function,
        )

        init_unit_parameters = np.zeros(shape=(self.config_dict["nlive"], model.prior_count))
        init_parameters = np.zeros(shape=(self.config_dict["nlive"], model.prior_count))
        init_log_likelihoods = np.zeros(shape=(self.config_dict["nlive"]))

        for index in range(len(parameters)):

            init_unit_parameters[index, :] = np.asarray(unit_parameters[index])
            init_parameters[index, :] = np.asarray(parameters[index])
            init_log_likelihoods[index] = np.asarray(log_likelihoods[index])

        return [init_unit_parameters, init_parameters, init_log_likelihoods]

    def remove_state_files(self):
        self.paths.remove_object("dynesty")


class DynestyStatic(AbstractDynesty):

    def __init__(
            self,
            name=None,
            path_prefix=None,
            prior_passer=None,
            iterations_per_update=None,
            number_of_cores=None,
            session=None,
            **kwargs
    ):
        """
        A Dynesty `NonLinearSearch` using a static number of live points.

        For a full description of Dynesty, checkout its GitHub and readthedocs webpages:

        https://github.com/joshspeagle/dynesty
        https://dynesty.readthedocs.io/en/latest/index.html

        Parameters
        ----------
        name : str
            The name of the search, controlling the last folder results are output.
        path_prefix : str
            The path of folders prefixing the name folder where results are output.
        prior_passer : af.PriorPasser
            Controls how priors are passed from the results of this `NonLinearSearch` to a subsequent non-linear search.
        iterations_per_update : int
            The number of iterations performed between every Dynesty back-up (via dumping the Dynesty instance as a
            pickle).
        number_of_cores : int
            The number of cores Emcee sampling is performed using a Python multiprocessing Pool instance. If 1, a
            pool instance is not created and the job runs in serial.
        """

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            prior_passer=prior_passer,
            iterations_per_update=iterations_per_update,
            number_of_cores=number_of_cores,
            session=session,
            **kwargs
        )

        logger.debug("Creating DynestyStatic NLO")

    def sampler_from(self, model, fitness_function, pool):
        """Get the static Dynesty sampler which performs the non-linear search, passing it all associated input Dynesty
        variables."""

        live_points = self.initial_live_points_from_model_and_fitness_function(
            model=model, fitness_function=fitness_function
        )

        return StaticSampler(
            loglikelihood=fitness_function,
            prior_transform=prior_transform,
            ndim=model.prior_count,
            logl_args=[model, fitness_function],
            ptform_args=[model],
            live_points=live_points,
            queue_size=self.number_of_cores,
            pool=pool,
            **self.config_dict
        )

    @property
    def config_dict_run_nested(self):

        config_dict = self.config_dict

        config_dict_run_nested = {}

        config_dict_run_nested["dlogz"] = config_dict["dlogz"]
        config_dict_run_nested["logl_max"] =  config_dict["logl_max"]
        config_dict_run_nested["n_effective"] =  config_dict["n_effective"]
        config_dict_run_nested["print_progress"] = not self.silence,

        return config_dict_run_nested


class DynestyDynamic(AbstractDynesty):
    def __init__(
            self,
            name=None,
            path_prefix=None,
            prior_passer=None,
            iterations_per_update=None,
            number_of_cores=None,
            **kwargs
    ):
        """
        A Dynesty non-linear search, using a dynamically changing number of live points.

        For a full description of Dynesty, checkout its GitHub and readthedocs webpages:

        https://github.com/joshspeagle/dynesty
        https://dynesty.readthedocs.io/en/latest/index.html

        Parameters
        ----------
        name : str
            The name of the search, controlling the last folder results are output.
        path_prefix : str
            The path of folders prefixing the name folder where results are output.
        prior_passer : af.PriorPasser
            Controls how priors are passed from the results of this `NonLinearSearch` to a subsequent non-linear search.
        iterations_per_update : int
            The number of iterations performed between every Dynesty back-up (via dumping the Dynesty instance as a
            pickle).
        number_of_cores : int
            The number of cores Emcee sampling is performed using a Python multiprocessing Pool instance. If 1, a
            pool instance is not created and the job runs in serial.
        """

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            prior_passer=prior_passer,
            iterations_per_update=iterations_per_update,
            number_of_cores=number_of_cores,
            **kwargs
        )

        logger.debug("Creating DynestyDynamic NLO")

    def sampler_from(self, model, fitness_function, pool):
        """
        Get the dynamic Dynesty sampler which performs the non-linear search, passing it all associated input Dynesty
        variables.
        """

        live_points = self.initial_live_points_from_model_and_fitness_function(
            model=model, fitness_function=fitness_function
        )

        return DynamicNestedSampler(
            loglikelihood=fitness_function,
            prior_transform=prior_transform,
            ndim=model.prior_count,
            logl_args=[model, fitness_function],
            ptform_args=[model],
            live_points=live_points,
            queue_size=self.number_of_cores,
            pool=pool,
            **self.config_dict
        )