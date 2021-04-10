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

class AbstractDynesty(AbstractNest, ABC):
    def __init__(
            self,
            name : str ="",
            path_prefix : str= "",
            prior_passer : PriorPasser = None,
            terminate_at_acceptance_ratio=None,
            acceptance_ratio_threshold=None,
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
        terminate_at_acceptance_ratio : bool
            If `True`, the sampler will automatically terminate when the acceptance ratio falls behind an input
            threshold value (see *Nest* for a full description of this feature).
        acceptance_ratio_threshold : float
            The acceptance ratio threshold below which sampling terminates if *terminate_at_acceptance_ratio* is
            `True` (see *Nest* for a full description of this feature).
        iterations_per_update : int
            The number of iterations performed between every Dynesty back-up (via dumping the Dynesty instance as a
            pickle).
        number_of_cores : int
            The number of cores Emcee sampling is performed using a Python multiprocessing Pool instance. If 1, a
            pool instance is not created and the job runs in serial.
        """

        self.kwargs = kwargs

        self.nlive = self.config_dict["nlive"]
        self.facc = self.config_dict["facc"]
        self.bound = self.config_dict["bound"]
        self.sample = self.config_dict["sample"]
        self.bootstrap = self.config_dict["bootstrap"]
        self.enlarge = self.config_dict["enlarge"]
        self.update_interval = self.config_dict["update_interval"]
        self.vol_dec = self.config_dict["vol_dec"]
        self.vol_check = self.config_dict["vol_check"]
        self.walks = self.config_dict["walks"]
        self.slices = self.config_dict["slices"]
        self.fmove = self.config_dict["fmove"]
        self.max_move = self.config_dict["max_move"]
        self.maxiter = self.config_dict["maxiter"]
        self.maxcall = self.config_dict["maxcall"]
        self.logl_max = self.config_dict["logl_max"]
        self.n_effective = self.config_dict["n_effective"]
        self.dlogz = self.config_dict["dlogz"]

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            prior_passer=prior_passer,
            terminate_at_acceptance_ratio=terminate_at_acceptance_ratio,
            acceptance_ratio_threshold=acceptance_ratio_threshold,
            iterations_per_update=iterations_per_update,
            session=session
        )

        self.number_of_cores = (
            self._config("parallel", "number_of_cores")
            if number_of_cores is None
            else number_of_cores
        )

        logger.debug("Creating DynestyStatic NLO")

    @property
    def no_limit(self):
        if self.maxcall is None:
            return True
        return False

    class Fitness(AbstractNest.Fitness):
        @property
        def resample_figure_of_merit(self):
            """If a sample raises a FitException, this value is returned to signify that the point requires resampling or
             should be given a likelihood so low that it is discard.

             -np.inf is an invalid sample value for Dynesty, so we instead use a large negative number."""
            return -1.0e99

    def _fit(self, model: AbstractPriorModel, analysis, log_likelihood_cap=None) -> Result:
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
            logger.info("Existing Dynesty samples found, resuming non-linear search.")

        else:

            sampler = self.sampler_fom_model_and_fitness(
                model=model, fitness_function=fitness_function, pool=pool
            )

            logger.info("No Dynesty samples found, beginning new non-linear search. ")

        finished = False

        while not finished:

            try:
                total_iterations = np.sum(sampler.results.ncall)
            except AttributeError:
                total_iterations = 0

            if not self.no_limit:
                iterations = self.maxcall - total_iterations
            else:
                iterations = self.iterations_per_update

            if iterations > 0:

                for i in range(10):

                    try:

                        sampler.run_nested(
                            maxcall=iterations,
                            dlogz=self.dlogz,
                            logl_max=self.logl_max,
                            n_effective=self.n_effective,
                            print_progress=not self.silence,
                        )

                        if i == 9:
                            raise ValueError("Dynesty crashed due to repeated bounding errors")

                        break

                    except (ValueError, np.linalg.LinAlgError):

                        continue

            sampler_pickle = sampler
            sampler_pickle.loglikelihood = None

            self.paths.save_object(
                "dynesty",
                sampler_pickle
            )

            sampler_pickle.loglikelihood = fitness_function

            self.perform_update(model=model, analysis=analysis, during_analysis=True)

            iterations_after_run = np.sum(sampler.results.ncall)

            if (
                    total_iterations == iterations_after_run
                    or total_iterations == self.maxcall
            ):
                finished = True

    def sampler_fom_model_and_fitness(self, model, fitness_function):
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
            number_live_points=sampler.results.nlive,
            time=self.timer.time
        )

    def initial_live_points_from_model_and_fitness_function(
            self, model, fitness_function
    ):

        unit_parameters, parameters, log_likelihoods = self.initializer.initial_samples_from_model(
            total_points=self.nlive,
            model=model,
            fitness_function=fitness_function,
        )

        init_unit_parameters = np.zeros(shape=(self.nlive, model.prior_count))
        init_parameters = np.zeros(shape=(self.nlive, model.prior_count))
        init_log_likelihoods = np.zeros(shape=(self.nlive))

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
            name="",
            path_prefix="",
            prior_passer=None,
            terminate_at_acceptance_ratio=None,
            acceptance_ratio_threshold=None,
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
        terminate_at_acceptance_ratio : bool
            If `True`, the sampler will automatically terminate when the acceptance ratio falls behind an input
            threshold value (see *Nest* for a full description of this feature).
        acceptance_ratio_threshold : float
            The acceptance ratio threshold below which sampling terminates if *terminate_at_acceptance_ratio* is
            `True` (see *Nest* for a full description of this feature).
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
            terminate_at_acceptance_ratio=terminate_at_acceptance_ratio,
            acceptance_ratio_threshold=acceptance_ratio_threshold,
            number_of_cores=number_of_cores,
            session=session,
            **kwargs
        )

        logger.debug("Creating DynestyStatic NLO")

    def sampler_fom_model_and_fitness(self, model, fitness_function, pool):
        """Get the static Dynesty sampler which performs the non-linear search, passing it all associated input Dynesty
        variables."""

        live_points = self.initial_live_points_from_model_and_fitness_function(
            model=model, fitness_function=fitness_function
        )

        return StaticSampler(
            loglikelihood=fitness_function,
            prior_transform=AbstractDynesty.Fitness.prior,
            ndim=model.prior_count,
            logl_args=[model, fitness_function],
            ptform_args=[model],
            live_points=live_points,
            queue_size=self.number_of_cores,
            pool=pool,
            **self.config_dict
        )


class DynestyDynamic(AbstractDynesty):
    def __init__(
            self,
            name="",
            path_prefix="",
            prior_passer=None,
            terminate_at_acceptance_ratio=None,
            acceptance_ratio_threshold=None,
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
        terminate_at_acceptance_ratio : bool
            If `True`, the sampler will automatically terminate when the acceptance ratio falls behind an input
            threshold value (see *Nest* for a full description of this feature).
        acceptance_ratio_threshold : float
            The acceptance ratio threshold below which sampling terminates if *terminate_at_acceptance_ratio* is
            `True` (see *Nest* for a full description of this feature).
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
            terminate_at_acceptance_ratio=terminate_at_acceptance_ratio,
            acceptance_ratio_threshold=acceptance_ratio_threshold,
            iterations_per_update=iterations_per_update,
            number_of_cores=number_of_cores,
            **kwargs
        )

        if self.nlive <= 0:
            self.nlive = 500

        logger.debug("Creating DynestyDynamic NLO")

    def sampler_fom_model_and_fitness(self, model, fitness_function):
        """Get the dynamic Dynesty sampler which performs the non-linear search, passing it all associated input Dynesty
        variables."""
        return DynamicNestedSampler(
            loglikelihood=fitness_function,
            prior_transform=AbstractDynesty.Fitness.prior,
            ndim=model.prior_count,
            logl_args=[model, fitness_function],
            ptform_args=[model],
            bound=self.bound,
            sample=self.sample,
            update_interval=self.update_interval,
            bootstrap=self.bootstrap,
            enlarge=self.enlarge,
            vol_dec=self.vol_dec,
            vol_check=self.vol_check,
            walks=self.walks,
            facc=self.facc,
            slices=self.slices,
            fmove=self.fmove,
            max_move=self.max_move,
        )

    def perform_update(self, model, analysis, during_analysis):
        """
        Perform an update of the `NonLinearSearch` results, which occurs every *iterations_per_update* of the
        non-linear search. The update performs the following tasks:

        1) Visualize the maximum log likelihood model.
        2) Output the model results to the model.reults file.

        These task are performed every n updates, set by the relevent *task_every_update* variable, for example
        *visualize_every_update*.

        Parameters
        ----------
        model : ModelMapper
            The model which generates instances for different points in parameter space.
        analysis : Analysis
            Contains the data and the log likelihood function which fits an instance of the model to the data, returning
            the log likelihood the `NonLinearSearch` maximizes.
        during_analysis : bool
            If the update is during a non-linear search, in which case tasks are only performed after a certain number
             of updates and only a subset of visualization may be performed.
        """
        pass

    def fit(self, model, analysis: "Analysis", info=None) -> "Result":
        """ Fit a model, M with some function f that takes instances of the
        class represented by model M and gives a score for their fitness.

        A model which represents possible instances with some dimensionality is fit.

        The analysis provides two functions. One visualises an instance of a model and the
        other scores an instance based on how well it fits some data. The search
        produces instances of the model by picking points in an N dimensional space.

        Parameters
        ----------
        analysis : af.Analysis
            An object that encapsulates the data and a log likelihood function.
        model : ModelMapper
            An object that represents possible instances of some model with a
            given dimensionality which is the number of free dimensions of the
            model.
        info : dict
            Optional dictionary containing information about the fit that can be loaded by the aggregator.

        Returns
        -------
        An object encapsulating how well the model fit the data, the best fit instance
        and an updated model with free parameters updated to represent beliefs
        produced by this fit.
        """

        self.paths.restore()
        self.setup_log_file()

        self.paths.save_all(
            info=info,
            pickle_files=[]
        )

        # TODO : Better way to handle?
        self.timer.samples_path = self.paths.samples_path
        self.timer.start()

        samples = self._fit(model=model, analysis=analysis)
        self.paths.completed()

        return Result(samples=samples, model=model, search=self)

    def _fit(self, model: AbstractPriorModel, analysis, log_likelihood_cap=None) -> NestSamples:
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
            model=model, analysis=analysis, pool_ids=pool_ids
        )

        sampler = self.sampler_fom_model_and_fitness(
            model=model, fitness_function=fitness_function
        )

        logger.info(
            "No DynestyDynamic samples found, beginning new non-linear search. "
        )

        finished = False

        while not finished:

            try:
                total_iterations = np.sum(sampler.results.ncall)
            except AttributeError:
                total_iterations = 0

            if not self.no_limit:
                iterations = self.maxcall - total_iterations
            else:
                iterations = self.iterations_per_update

            if iterations > 0:

                sampler.run_nested(
                    nlive_init=self.nlive,
                    maxcall=iterations,
                    dlogz_init=self.dlogz,
                    logl_max_init=self.logl_max,
                    n_effective=self.n_effective,
                    print_progress=not self.silence,
                )

            iterations_after_run = np.sum(sampler.results.ncall)

            if (
                    total_iterations == iterations_after_run
                    or total_iterations == self.maxcall
            ):
                finished = True

        during_analysis = False

        self.timer.update()

        samples = self.samples_via_sampler_from_model(model=model, sampler=sampler)

        self.paths.save_samples(samples)

        instance = samples.max_log_likelihood_instance

        if self.should_visualize() or not during_analysis:
            analysis.visualize(instance=instance, during_analysis=during_analysis)

        if self.should_output_model_results() or not during_analysis:

            self.paths.save_summary(
                samples=samples,
                log_likelihood_function_time=-1
            )

        self.paths.zip_remove()

        return samples

    def samples_via_sampler_from_model(self, model, sampler):
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
                log_likelihoods=log_likelihoods,
                log_priors=log_priors,
                weights=weights,
                model=model
            ),
            total_samples=total_samples,
            log_evidence=log_evidence,
            number_live_points=self.nlive,
            time=self.timer.time,
        )
