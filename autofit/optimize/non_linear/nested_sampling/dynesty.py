import logging
import os
import pickle
import numpy as np
from dynesty import NestedSampler as StaticSampler
from dynesty.dynesty import DynamicNestedSampler
from multiprocessing.pool import Pool

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.optimize.non_linear import samples
from autofit.optimize.non_linear.nested_sampling.nested_sampler import NestedSampler
from autofit.optimize.non_linear import non_linear as nl
from autofit.optimize.non_linear.non_linear import Result
from autofit.text import samples_text

logger = logging.getLogger(__name__)



class AbstractDynesty(NestedSampler):
    def __init__(
        self,
        paths=None,
        sigma=3,
        bound=None,
        sample=None,
        bootstrap=None,
        enlarge=None,
        update_interval=None,
        vol_dec=None,
        vol_check=None,
        walks=None,
        facc=None,
        slices=None,
        fmove=None,
        max_move=None,
        terminate_at_acceptance_ratio=None,
        acceptance_ratio_threshold=None,
        iterations_per_update=None,
        number_of_cores=None,
    ):
        """
        Class to setup and run a Dynesty non-linear search.

        For a full description of Dynesty, checkout its GitHub and readthedocs webpages:

        https://github.com/joshspeagle/dynesty

        https://dynesty.readthedocs.io/en/latest/index.html

        **PyAutoFit** extends Dynesty by allowing runs to be terminated and resumed from that point. This is achieved
        by pickling the sampler instance during the model-fit after an input number of iterations.

        Attributes unique to **PyAutoFit** are described below, all remaining attributes are DyNesty parameters are
        described at the Dynesty API webpage:

        https://dynesty.readthedocs.io/en/latest/api.html#dynesty.dynamicsampler.stopping_function

        Parameters
        ----------
        paths : af.Paths
            A class that manages all paths, e.g. where the phase outputs are stored, the non-linear search samples,
            backups, etc.
        sigma : float
            The error-bound value that linked Gaussian prior withs are computed using. For example, if sigma=3.0,
            parameters will use Gaussian Priors with widths coresponding to errors estimated at 3 sigma confidence.
        terminate_at_acceptance_ratio : bool
            If *True*, the sampler will automatically terminate when the acceptance ratio falls behind an input
            threshold value (see *NestedSampler* for a full description of this feature).
        acceptance_ratio_threshold : float
            The acceptance ratio threshold below which sampling terminates if *terminate_at_acceptance_ratio* is
            *True* (see *NestedSampler* for a full description of this feature).
        number_of_cores : int
            The number of cores Emcee sampling is performed using a Python multiprocessing Pool instance. If 1, a
            pool instance is not created and the job runs in serial.

        Attributes
        ----------
        sigma : float
            The error-bound value that linked Gaussian prior withs are computed using. For example, if sigma=3.0,
            parameters will use Gaussian Priors with widths coresponding to errors estimated at 3 sigma confidence.
        iterations_per_update : int
            The number of iterations performed between every Dynesty back-up (via dumping the Dynesty instance as a
            pickle).
        """

        super().__init__(
            paths=paths,
            sigma=sigma,
            terminate_at_acceptance_ratio=terminate_at_acceptance_ratio,
            acceptance_ratio_threshold=acceptance_ratio_threshold,
        )

        self.bound = self.config("search", "bound", str) if bound is None else bound
        self.sample = self.config("search", "sample", str) if sample is None else sample
        self.bootstrap = (
            self.config("search", "bootstrap", int) if bootstrap is None else bootstrap
        )
        self.enlarge = self.config("search", "enlarge", float) if enlarge is None else enlarge

        self.update_interval = (
            self.config("search", "update_interval", float)
            if update_interval is None
            else update_interval
        )

        if self.update_interval < 0.0:
            self.update_interval = None

        if self.enlarge < 0.0:
            if self.bootstrap == 0.0:
                self.enlarge = 1.0
            else:
                self.enlarge = 1.25

        self.vol_dec = self.config("search", "vol_dec", float) if vol_dec is None else vol_dec
        self.vol_check = (
            self.config("search", "vol_check", float) if vol_check is None else vol_check
        )
        self.walks = self.config("search", "walks", int) if walks is None else walks
        self.facc = self.config("search", "facc", float) if facc is None else facc
        self.slices = self.config("search", "slices", int) if slices is None else slices
        self.fmove = self.config("search", "fmove", float) if fmove is None else fmove
        self.max_move = self.config("search", "max_move", int) if max_move is None else max_move

        self.iterations_per_update = (
            self.config("settings", "iterations_per_update", int)
            if iterations_per_update is None
            else iterations_per_update
        )

        self.number_of_cores = (
            self.config("parallel", "number_of_cores", int)
            if number_of_cores is None
            else number_of_cores
        )

        logger.debug("Creating DynestyStatic NLO")

    def copy_with_name_extension(self, extension, remove_phase_tag=False):
        """Copy this instance of the dynesty non-linear search with all associated attributes.

        This is used to set up the non-linear search on phase extensions."""
        copy = super().copy_with_name_extension(
            extension=extension, remove_phase_tag=remove_phase_tag
        )
        copy.iterations_per_update = self.iterations_per_update
        copy.bound = self.bound
        copy.sample = self.sample
        copy.update_interval = self.update_interval
        copy.bootstrap = self.bootstrap
        copy.enlarge = self.enlarge
        copy.vol_dec = self.vol_dec
        copy.vol_check = self.vol_check
        copy.walks = self.walks
        copy.facc = self.facc
        copy.slices = self.slices
        copy.fmove = self.fmove
        copy.max_move = self.max_move
        copy.number_of_cores = self.number_of_cores

        return copy

    def _fit(self, model: AbstractPriorModel, analysis) -> Result:
        """
        Fit a model using Dynesty and a function that returns a log likelihood from instances of that model.

        Dynesty is not called once, but instead called multiple times every iterations_per_update, such that the
        sampler instance can be pickled during the model-fit. This allows runs to be terminated and resumed.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space. This maps the points from unit
            cube values to physical values via the priors.
        fitness_function
            A function that fits this model to the data, returning the log likelihood of the fit.

        Returns
        -------
        A result object comprising the best-fit model instance, log_likelihood and an *Output* class that enables analysis
        of the full samples used by the fit.
        """

        pool, pool_ids = self.make_pool()

        fitness_function = self.fitness_function_from_model_and_analysis(
            model=model, analysis=analysis
        )

        if os.path.exists("{}/{}.pickle".format(self.paths.samples_path, "dynesty")):

            sampler = self.load_sampler

        else:

            try:
                os.makedirs(self.paths.samples_path)
            except FileExistsError:
                pass

            sampler = self.sampler_fom_model_and_fitness(
                model=model, fitness_function=fitness_function
            )

        # These hacks are necessary to be able to pickle the sampler.

        sampler.rstate = np.random
        sampler.pool = pool

        if self.number_of_cores == 1:
            sampler.M = map
        else:
            sampler.M = pool.map

        dynesty_finished = False

        while not dynesty_finished:

            try:
                iterations_before_run = np.sum(sampler.results.ncall)
            except AttributeError:
                iterations_before_run = 0

            sampler.run_nested(maxcall=self.iterations_per_update)

            iterations_after_run = np.sum(sampler.results.ncall)

            with open(f"{self.paths.samples_path}/dynesty.pickle", "wb") as f:
                pickle.dump(sampler, f)

            if iterations_before_run == iterations_after_run:

                dynesty_finished = True

        self.paths.backup()

        samples = self.samples_from_model(model=model)

        samples_text.results_to_file(
            samples=samples, file_results=self.paths.file_results, during_analysis=False
        )

        return Result(samples=samples, previous_model=model)

    @property
    def load_sampler(self):
        with open("{}/{}.pickle".format(self.paths.samples_path, "dynesty"), "rb") as f:
            return pickle.load(f)

    def sampler_fom_model_and_fitness(self, model, fitness_function):
        return NotImplementedError()

    def samples_from_model(self, model):
        """Create a *Samples* object from this non-linear search's output files on the hard-disk and model.

        For Dynesty, all information that we need is available from the instance of the dynesty sampler.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space. This maps the points from unit
            cube values to physical values via the priors.
        paths : af.Paths
            A class that manages all paths, e.g. where the phase outputs are stored, the non-linear search samples,
            backups, etc.
        """

        sampler = self.load_sampler

        parameters = sampler.results.samples
        log_priors = [
            sum(model.log_priors_from_vector(vector=vector)) for vector in parameters
        ]
        log_likelihoods = sampler.results.logl
        weights = sampler.results.logwt
        total_samples = int(np.sum(sampler.results.ncall))
        log_evidence = np.max(sampler.results.logz)

        return samples.NestedSamplerSamples(
            model=model,
            parameters=parameters,
            log_likelihoods=log_likelihoods,
            log_priors=log_priors,
            weights=weights,
            total_samples=total_samples,
            log_evidence=log_evidence,
            number_live_points=sampler.results.nlive,
        )


class DynestyStatic(AbstractDynesty):
    def __init__(
        self,
        paths=None,
        sigma=3,
        n_live_points=None,
        bound=None,
        sample=None,
        bootstrap=None,
        enlarge=None,
        update_interval=None,
        vol_dec=None,
        vol_check=None,
        walks=None,
        facc=None,
        slices=None,
        fmove=None,
        max_move=None,
        terminate_at_acceptance_ratio=None,
        acceptance_ratio_threshold=None,
        iterations_per_update=None,
        number_of_cores=None,
    ):
        """
        Class to setup and run a Dynesty non-linear search, using the static Dynesty nested sampler described at this
        webpage:

        https://dynesty.readthedocs.io/en/latest/api.html#module-dynesty.dynesty

        See the docstring for *AbstractDynesty* for complete documentation of Dynesty samplers.

        Attributes
        ----------
        sigma : float
            The error-bound value that linked Gaussian prior withs are computed using. For example, if sigma=3.0,
            parameters will use Gaussian Priors with widths coresponding to errors estimated at 3 sigma confidence.
        iterations_per_update : int
            The number of iterations performed between every Dynesty back-up (via dumping the Dynesty instance as a
            pickle).
        """

        super().__init__(
            paths=paths,
            sigma=sigma,
            iterations_per_update=iterations_per_update,
            bound=bound,
            sample=sample,
            bootstrap=bootstrap,
            enlarge=enlarge,
            update_interval=update_interval,
            vol_dec=vol_dec,
            vol_check=vol_check,
            walks=walks,
            facc=facc,
            slices=slices,
            fmove=fmove,
            max_move=max_move,
            terminate_at_acceptance_ratio=terminate_at_acceptance_ratio,
            acceptance_ratio_threshold=acceptance_ratio_threshold,
            number_of_cores=number_of_cores,
        )

        self.n_live_points = (
            self.config("search", "n_live_points", int)
            if n_live_points is None
            else n_live_points
        )

        logger.debug("Creating DynestyStatic NLO")

    def copy_with_name_extension(self, extension, remove_phase_tag=False):
        """Copy this instance of the dynesty non-linear search with all associated attributes.

        This is used to set up the non-linear search on phase extensions."""
        copy = super().copy_with_name_extension(
            extension=extension, remove_phase_tag=remove_phase_tag
        )
        copy.n_live_points = self.n_live_points

        return copy

    def sampler_fom_model_and_fitness(self, model, fitness_function):
        """Get the static Dynesty sampler which performs the non-linear search, passing it all associated input Dynesty
        variables."""

        return StaticSampler(
            loglikelihood=fitness_function,
            prior_transform=nl.NonLinearOptimizer.Fitness.prior,
            ndim=model.prior_count,
            logl_args=[model, fitness_function],
            ptform_args=[model],
            nlive=self.n_live_points,
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


class DynestyDynamic(AbstractDynesty):
    def __init__(
        self,
        paths=None,
        sigma=3,
        iterations_per_update=None,
        bound=None,
        sample=None,
        bootstrap=None,
        enlarge=None,
        update_interval=None,
        vol_dec=None,
        vol_check=None,
        walks=None,
        facc=None,
        slices=None,
        fmove=None,
        max_move=None,
        terminate_at_acceptance_ratio=None,
        acceptance_ratio_threshold=None,
        number_of_cores=None,
    ):
        """
        Class to setup and run a Dynesty non-linear search, using the dynamic Dynesty nested sampler described at this
        webpage:

        https://dynesty.readthedocs.io/en/latest/api.html#module-dynesty.dynamicsampler

        See the docstring for *AbstractDynesty* for complete documentation of Dynesty samplers.

        Attributes
        ----------
        sigma : float
            The error-bound value that linked Gaussian prior withs are computed using. For example, if sigma=3.0,
            parameters will use Gaussian Priors with widths coresponding to errors estimated at 3 sigma confidence.
        iterations_per_update : int
            The number of iterations performed between every Dynesty back-up (via dumping the Dynesty instance as a
            pickle).
        """

        super().__init__(
            paths=paths,
            sigma=sigma,
            iterations_per_update=iterations_per_update,
            bound=bound,
            sample=sample,
            bootstrap=bootstrap,
            enlarge=enlarge,
            update_interval=update_interval,
            vol_dec=vol_dec,
            vol_check=vol_check,
            walks=walks,
            facc=facc,
            slices=slices,
            fmove=fmove,
            max_move=max_move,
            terminate_at_acceptance_ratio=terminate_at_acceptance_ratio,
            acceptance_ratio_threshold=acceptance_ratio_threshold,
            number_of_cores=number_of_cores,
        )

        logger.debug("Creating DynestyDynamic NLO")

    def sampler_fom_model_and_fitness(self, model, fitness_function):
        """Get the dynamic Dynesty sampler which performs the non-linear search, passing it all associated input Dynesty
        variables."""
        return DynamicNestedSampler(
            loglikelihood=fitness_function,
            prior_transform=nl.NonLinearOptimizer.Fitness.prior,
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
