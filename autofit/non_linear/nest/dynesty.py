from os import path
from abc import ABC

import numpy as np

from typing import Optional

from dynesty import NestedSampler as StaticSampler
from dynesty.dynesty import DynamicNestedSampler
from dynesty.results import Results

from autoconf import conf

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.log import logger
from autofit.non_linear.abstract_search import PriorPasser
from autofit.non_linear.nest.abstract_nest import AbstractNest
from autofit.non_linear.samples import NestSamples, Sample

from autofit.plot import DynestyPlotter
from autofit.plot.mat_wrap.wrap.wrap_base import Output

def prior_transform(cube, model):
    phys_cube = model.vector_from_unit_vector(unit_vector=cube)

    for i in range(len(phys_cube)):
        cube[i] = phys_cube[i]

    return cube


class AbstractDynesty(AbstractNest, ABC):

    def __init__(
            self,
            name: str = "",
            path_prefix: str = "",
            unique_tag : Optional[str] = None,
            prior_passer: PriorPasser = None,
            iterations_per_update : int = None,
            number_of_cores : int = None,
            session : Optional[bool] = None,
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

        logger.debug("Creating DynestyStatic Search")

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

                        dihfoisdj

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

    def sampler_from(self, model, fitness_function, pool):
        return NotImplementedError()

    def samples_from(self, model):
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

        try:

            sampler = self.paths.load_object(
                "dynesty"
            )
            results = sampler.results

        except FileNotFoundError:

            samples = self.paths.load_object(
                "samples"
            )
            results = samples.results

        return DynestySamples(
            model=model,
            results=results,
            number_live_points=self.total_live_points,
            unconverged_sample_size=1,
            time=self.timer.time,
        )

    def initial_live_points_from_model_and_fitness_function(
            self, model, fitness_function
    ):

        unit_parameters, parameters, log_likelihood_list = self.initializer.initial_samples_from_model(
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
        

class DynestyStatic(AbstractDynesty):

    __identifier_fields__ = (
        "nlive",
        "bound",
        "sample",
        "enlarge",
        "bootstrap",
        "vol_dec",
        "walks",
        "facc",
        "slices",
        "fmove",
        "max_move"
    )

    def __init__(
            self,
            name=None,
            path_prefix=None,
            unique_tag : Optional[str] = None,
            prior_passer=None,
            iterations_per_update : int = None,
            number_of_cores : int = None,
            session : Optional[bool] = None,
            **kwargs
    ):
        """
        A Dynesty `NonLinearSearch` using a static number of live points.

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

        logger.debug("Creating DynestyStatic Search")

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
            **self.config_dict_search
        )

    @property
    def total_live_points(self):
        return self.config_dict_search["nlive"]


class DynestyDynamic(AbstractDynesty):

    __identifier_fields__ = (
        "bound",
        "sample",
        "enlarge",
        "bootstrap",
        "vol_dec",
        "walks",
        "facc",
        "slices",
        "fmove",
        "max_move"
    )

    def __init__(
            self,
            name=None,
            path_prefix=None,
            unique_tag : Optional[str] = None,
            prior_passer=None,
            iterations_per_update : int = None,
            number_of_cores : int = None,
            **kwargs
    ):
        """
        A Dynesty non-linear search, using a dynamically changing number of live points.

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
        """

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            prior_passer=prior_passer,
            iterations_per_update=iterations_per_update,
            number_of_cores=number_of_cores,
            **kwargs
        )

        logger.debug("Creating DynestyDynamic Search")

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
            **self.config_dict_search
        )

    @property
    def total_live_points(self):
        return self.config_dict_run["nlive_init"]


class DynestySamples(NestSamples):

    def __init__(
            self,
            model: AbstractPriorModel,
            results: Results,
            number_live_points : int,
            unconverged_sample_size: int = 100,
            time: float = None,
    ):
        """
        The *Output* classes in **PyAutoFit** provide an interface between the results of a `NonLinearSearch` (e.g.
        as files on your hard-disk) and Python.

        For example, the output class can be used to load an instance of the best-fit model, get an instance of any
        individual sample by the `NonLinearSearch` and return information on the likelihoods, errors, etc.

        The Bayesian log evidence estimated by the nested sampling algorithm.

        Parameters
        ----------
        model : af.ModelMapper
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        number_live_points : int
            The number of live points used by the nested sampler.
        log_evidence : float
            The log of the Bayesian evidence estimated by the nested sampling algorithm.
        """

        super().__init__(
            model=model,
            unconverged_sample_size=unconverged_sample_size,
            time=time,
        )

        self.results = results
        self._samples = None
        self._number_live_points = number_live_points

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

        parameter_lists = self.results.samples.tolist()
        log_prior_list = [
            sum(self.model.log_prior_list_from_vector(vector=vector)) for vector in parameter_lists
        ]
        log_likelihood_list = list(self.results.logl)

        try:
            weight_list = list(
                np.exp(np.asarray(self.results.logwt) - self.results.logz[-1])
            )
        except:
            weight_list = self.results["weights"]

        self._samples = Sample.from_lists(
            model=self.model,
            parameter_lists=parameter_lists,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list
        )

        return self._samples

    @property
    def number_live_points(self):
        return self._number_live_points

    @property
    def total_samples(self):
        return int(np.sum(self.results.ncall))

    @property
    def log_evidence(self):
        return np.max(self.results.logz)