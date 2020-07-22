import os
import sys
import pickle
import numpy as np
from dynesty import NestedSampler as StaticSampler
from dynesty.dynesty import DynamicNestedSampler

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.samples import NestSamples
from autofit.non_linear.nest.abstract_nest import AbstractNest
from autofit.non_linear.abstract_search import Result
from autofit.text import samples_text
from autofit import exc
from autofit.non_linear.log import logger


class AbstractDynesty(AbstractNest):
    def __init__(
        self,
        paths=None,
        prior_passer=None,
        n_live_points=None,
        facc=None,
        evidence_tolerance=None,
        bound=None,
        sample=None,
        bootstrap=None,
        enlarge=None,
        update_interval=None,
        vol_dec=None,
        vol_check=None,
        walks=None,
        slices=None,
        fmove=None,
        max_move=None,
        maxiter=None,
        maxcall=None,
        logl_max=None,
        n_effective=None,
        terminate_at_acceptance_ratio=None,
        acceptance_ratio_threshold=None,
        iterations_per_update=None,
        number_of_cores=None,
    ):
        """
        A Dynesty non-linear search.

        For a full description of Dynesty, checkout its GitHub and readthedocs webpages:

        https://github.com/joshspeagle/dynesty

        https://dynesty.readthedocs.io/en/latest/index.html

        Extensions:

        - Allows runs to be terminated and resumed from the point it was terminated. This is achieved by pickling the
          sampler instance during the model-fit after an input number of iterations.

        Attributes unique to **PyAutoFit** are described below, all remaining attributes are DyNesty parameters are
        described at the Dynesty API webpage:

        https://dynesty.readthedocs.io/en/latest/api.html#dynesty.dynamicsampler.stopping_function

        Parameters
        ----------
        paths : af.Paths
            Manages all paths, e.g. where the search outputs are stored, the samples, backups, etc.
        prior_passer : af.PriorPasser
            Controls how priors are passed from the results of this non-linear search to a subsequent non-linear search.
        facc : float
            The target acceptance fraction for the 'rwalk' sampling option. Default is 0.5. Bounded to be between
            [1. / walks, 1.].
        evidence_threshold : float
            This is called dlogz in the Dynesty API. Iteration will stop when the estimated contribution of the 
            remaining prior volume to the total evidence falls below this threshold. Explicitly, the stopping 
            criterion is ln(z + z_est) - ln(z) < dlogz, where z is the current evidence from all saved samples and 
            z_est is the estimated contribution from the remaining volume. If add_live is True, the default is 
            1e-3 * (nlive - 1) + 0.01. Otherwise, the default is 0.01.
        bound : str
            Method used to approximately bound the prior using the current set of live points. Conditions the sampling
            methods used to propose new live points. Choices are no bound ('none'), a single bounding ellipsoid
            ('single'), multiple bounding ellipsoids ('multi'), balls centered on each live point ('balls'), and cubes
            centered on each live point ('cubes'). Default is 'multi'.
        samples : str
            Method used to sample uniformly within the likelihood constraint, conditioned on the provided bounds.
            Unique methods available are: uniform sampling within the bounds('unif'), random walks with fixed
            proposals ('rwalk'), random walks with variable (“staggering”) proposals ('rstagger'), multivariate slice
            sampling along preferred orientations ('slice'), “random” slice sampling along all orientations ('rslice'),
            “Hamiltonian” slices along random trajectories ('hslice'), and any callable function which follows the
            pattern of the sample methods defined in dynesty.sampling. 'auto' selects the sampling method based on the
            dimensionality of the problem (from ndim). When ndim < 10, this defaults to 'unif'. When 10 <= ndim <= 20,
            this defaults to 'rwalk'. When ndim > 20, this defaults to 'hslice' if a gradient is provided and 'slice'
            otherwise. 'rstagger' and 'rslice' are provided as alternatives for 'rwalk' and 'slice', respectively.
            Default is 'auto'.
        bootstrap : int
            Compute this many bootstrapped realizations of the bounding objects. Use the maximum distance found to the
            set of points left out during each iteration to enlarge the resulting volumes. Can lead to unstable
            bounding ellipsoids. Default is 0 (no bootstrap).
        enlarge : float
            Enlarge the volumes of the specified bounding object(s) by this fraction. The preferred method is to
            determine this organically using bootstrapping. If bootstrap > 0, this defaults to 1.0. If bootstrap = 0,
            this instead defaults to 1.25.
        vol_dec : float
            For the 'multi' bounding option, the required fractional reduction in volume after splitting an ellipsoid
            in order to to accept the split. Default is 0.5.
        vol_check : float
            For the 'multi' bounding option, the factor used when checking if the volume of the original bounding
            ellipsoid is large enough to warrant > 2 splits via ell.vol > vol_check * nlive * pointvol. Default is 2.0.
        walks : int
            For the 'rwalk' sampling option, the minimum number of steps (minimum 2) before proposing a new live point.
            Default is 25.
        update_interval : int or float
            If an integer is passed, only update the proposal distribution every update_interval-th likelihood call.
            If a float is passed, update the proposal after every round(update_interval * nlive)-th likelihood call.
            Larger update intervals larger can be more efficient when the likelihood function is quick to evaluate.
            Default behavior is to target a roughly constant change in prior volume, with 1.5 for 'unif', 0.15 * walks
            for 'rwalk' and 'rstagger', 0.9 * ndim * slices for 'slice', 2.0 * slices for 'rslice', and 25.0 * slices
            for 'hslice'.
        slices : int
            For the 'slice', 'rslice', and 'hslice' sampling options, the number of times to execute a “slice update”
            before proposing a new live point. Default is 5. Note that 'slice' cycles through all dimensions when
            executing a “slice update”.
        fmove : float
            The target fraction of samples that are proposed along a trajectory (i.e. not reflecting) for the 'hslice'
            sampling option. Default is 0.9.
        max_move : int
            The maximum number of timesteps allowed for 'hslice' per proposal forwards and backwards in time.
            Default is 100.
        maxiter : int
            Maximum number of iterations. Iteration may stop earlier if the termination condition is reached. Default
            is sys.maxsize (no limit).
        maxcall : int
            Maximum number of likelihood evaluations. Iteration may stop earlier if termination condition is reached.
            Default is sys.maxsize (no limit).
        logl_max : float
            Iteration will stop when the sampled ln(likelihood) exceeds the threshold set by logl_max. Default is no
            bound (np.inf).
        n_effective : int
            Minimum number of effective posterior samples. If the estimated “effective sample size” (ESS) exceeds 
            this number, sampling will terminate. Default is no ESS (np.inf).
        terminate_at_acceptance_ratio : bool
            If *True*, the sampler will automatically terminate when the acceptance ratio falls behind an input
            threshold value (see *Nest* for a full description of this feature).
        acceptance_ratio_threshold : float
            The acceptance ratio threshold below which sampling terminates if *terminate_at_acceptance_ratio* is
            *True* (see *Nest* for a full description of this feature).
        iterations_per_update : int
            The number of iterations performed between every Dynesty back-up (via dumping the Dynesty instance as a
            pickle).
        number_of_cores : int
            The number of cores Emcee sampling is performed using a Python multiprocessing Pool instance. If 1, a
            pool instance is not created and the job runs in serial.
        """

        self.n_live_points = (
            self._config("search", "n_live_points", int)
            if n_live_points is None
            else n_live_points
        )

        self.evidence_tolerance = evidence_tolerance

        self.facc = (
            self._config("search", "sampling_efficiency", float)
            if facc is None
            else facc
        )

        self.bound = self._config("search", "bound", str) if bound is None else bound
        self.sample = self._config("search", "sample", str) if sample is None else sample
        self.bootstrap = (
            self._config("search", "bootstrap", int) if bootstrap is None else bootstrap
        )
        self.enlarge = (
            self._config("search", "enlarge", float) if enlarge is None else enlarge
        )

        self.update_interval = (
            self._config("search", "update_interval", float)
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

        self.vol_dec = (
            self._config("search", "vol_dec", float) if vol_dec is None else vol_dec
        )
        self.vol_check = (
            self._config("search", "vol_check", float)
            if vol_check is None
            else vol_check
        )
        self.walks = self._config("search", "walks", int) if walks is None else walks
        self.slices = self._config("search", "slices", int) if slices is None else slices
        self.fmove = self._config("search", "fmove", float) if fmove is None else fmove
        self.max_move = (
            self._config("search", "max_move", int) if max_move is None else max_move
        )

        self.maxiter = (
            self._config("search", "maxiter", int) if maxiter is None else maxiter
        )
        if self.maxiter <= 0:
            self.maxiter = sys.maxsize
        self.maxcall = (
            self._config("search", "maxcall", int) if maxcall is None else maxcall
        )
        self.no_limit = False
        if self.maxcall <= 0:
            self.maxcall = sys.maxsize
            self.no_limit = True
        self.logl_max = (
            self._config("search", "logl_max", float) if logl_max is None else logl_max
        )
        self.n_effective = (
            self._config("search", "n_effective", int)
            if n_effective is None
            else n_effective
        )
        if self.n_effective <= 0:
            self.n_effective = np.inf

        super().__init__(
            paths=paths,
            prior_passer=prior_passer,
            terminate_at_acceptance_ratio=terminate_at_acceptance_ratio,
            acceptance_ratio_threshold=acceptance_ratio_threshold,
            iterations_per_update=iterations_per_update,
        )

        self.number_of_cores = (
            self._config("parallel", "number_of_cores", int)
            if number_of_cores is None
            else number_of_cores
        )

        logger.debug("Creating DynestyStatic NLO")

    class Fitness(AbstractNest.Fitness):
        @property
        def resample_figure_of_merit(self):
            """If a sample raises a FitException, this value is returned to signify that the point requires resampling or
             should be given a likelihood so low that it is discard.

             -np.inf is an invalid sample value for Dynesty, so we instead use a large negative number."""
            return -1.0e99

    def _fit(self, model: AbstractPriorModel, analysis) -> Result:
        """
        Fit a model using Dynesty and the Analysis class which contains the data and returns the log likelihood from
        instances of the model, which the non-linear search seeks to maximize.

        Parameters
        ----------
        model : ModelMapper
            The model which generates instances for different points in parameter space.
        analysis : Analysis
            Contains the data and the log likelihood function which fits an instance of the model to the data, returning
            the log likelihood the non-linear search maximizes.

        Returns
        -------
        A result object comprising the Samples object that includes the maximum log likelihood instance and full
        set of accepted ssamples of the fit.
        """

        pool, pool_ids = self.make_pool()

        fitness_function = self.fitness_function_from_model_and_analysis(
            model=model, analysis=analysis
        )

        if os.path.exists("{}/{}.pickle".format(self.paths.samples_path, "dynesty")):

            sampler = self.load_sampler
            sampler.loglikelihood = fitness_function
            logger.info("Existing Dynesty samples found, resuming non-linear search.")

        else:

            try:
                os.makedirs(self.paths.samples_path)
            except FileExistsError:
                pass

            sampler = self.sampler_fom_model_and_fitness(
                model=model, fitness_function=fitness_function
            )

            logger.info("No Dynesty samples found, beginning new non-linear search. ")

        # These hacks are necessary to be able to pickle the sampler.

        sampler.rstate = np.random
        sampler.pool = pool

        if self.number_of_cores == 1:
            sampler.M = map
        else:
            sampler.M = pool.map

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
                    maxcall=iterations,
                    dlogz=self.evidence_tolerance,
                    logl_max=self.logl_max,
                    n_effective=self.n_effective,
                    print_progress=not self.silence,
                )

            sampler_pickle = sampler
            sampler_pickle.loglikelihood = None

            with open(f"{self.paths.samples_path}/dynesty.pickle", "wb") as f:
                pickle.dump(sampler_pickle, f)

            sampler_pickle.loglikelihood = fitness_function

            self.perform_update(model=model, analysis=analysis, during_analysis=True)

            iterations_after_run = np.sum(sampler.results.ncall)

            if (
                total_iterations == iterations_after_run
                or total_iterations == self.maxcall
            ):
                finished = True

    def copy_with_name_extension(self, extension, remove_phase_tag=False):
        """Copy this instance of the dynesty non-linear search with all associated attributes.

        This is used to set up the non-linear search on phase extensions."""
        copy = super().copy_with_name_extension(
            extension=extension, remove_phase_tag=remove_phase_tag
        )
        copy.n_live_points = self.n_live_points
        copy.iterations_per_update = self.iterations_per_update
        copy.evidence_tolerance = self.evidence_tolerance
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
        copy.initializer = self.initializer
        copy.iterations_per_update = self.iterations_per_update
        copy.number_of_cores = self.number_of_cores
        copy.terminate_at_acceptance_ratio = self.terminate_at_acceptance_ratio
        copy.acceptance_ratio_threshold = self.acceptance_ratio_threshold
        copy.stagger_resampling_likelihood = self.stagger_resampling_likelihood

        return copy

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
            Manages all paths, e.g. where the search outputs are stored, the samples, backups, etc.
        """

        sampler = self.load_sampler
        parameters = sampler.results.samples.tolist()
        log_priors = [
            sum(model.log_priors_from_vector(vector=vector)) for vector in parameters
        ]
        log_likelihoods = list(sampler.results.logl)

        try:
            weights = list(np.exp(np.asarray(sampler.results.logwt) - sampler.results.logz[-1]))
        except:
            weights = sampler.results['weights']

        total_samples = int(np.sum(sampler.results.ncall))
        log_evidence = np.max(sampler.results.logz)

        return NestSamples(
            model=model,
            parameters=parameters,
            log_likelihoods=log_likelihoods,
            log_priors=log_priors,
            weights=weights,
            total_samples=total_samples,
            log_evidence=log_evidence,
            number_live_points=sampler.results.nlive,
            time=self.timer.time
        )

    @property
    def tag(self):
        """Tag the output folder of the PySwarms non-linear search, according to the number of particles and
        parameters defining the search strategy."""

        name_tag = self._config("tag", "name", str)
        n_live_points_tag = (
            f"{self._config('tag', 'n_live_points')}_{self.n_live_points}"
        )

        sample_tag = f"{self._config('tag', 'sample')}_{self.sample}"
        bound_tag = f"{self._config('tag', 'bound')}_{self.bound}"
        vol_dec_tag = f"{self._config('tag', 'vol_dec')}_{self.vol_dec}"
        vol_check_tag = f"{self._config('tag', 'vol_check')}_{self.vol_check}"
        enlarge_tag = f"{self._config('tag', 'enlarge')}_{self.enlarge}"

        if self.bound in "multi":
            bound_multi_tag = f"{vol_dec_tag}_{vol_check_tag}"
            bound_tag = f"{bound_tag}_{bound_multi_tag}"

        if self.sample in "auto":
            return f"{name_tag}__{n_live_points_tag}__{bound_tag}__{enlarge_tag}__{sample_tag}"

        walks_tag = f"{self._config('tag', 'walks')}_{self.walks}"
        facc_tag = f"{self._config('tag', 'facc')}_{self.facc}"
        slices_tag = f"{self._config('tag', 'slices')}_{self.slices}"
        max_move_tag = f"{self._config('tag', 'max_move')}_{self.max_move}"

        if self.sample in "rwalk":
            method_tag = f"_{walks_tag}_{facc_tag}"
        elif self.sample == "hslice":
            method_tag = f"_{slices_tag}_{max_move_tag}"
        elif self.sample in ["slice", "rslice"]:
            method_tag = f"_{slices_tag}"
        else:
            method_tag = ""

        dynesty_tag = f"{bound_tag}__{enlarge_tag}__{sample_tag}{method_tag}"

        return f"{name_tag}__{n_live_points_tag}__{dynesty_tag}"

    def initial_live_points_from_model_and_fitness_function(
        self, model, fitness_function
    ):

        unit_parameters, parameters, log_likelihoods = self.initializer.initial_samples_from_model(
            total_points=self.n_live_points,
            model=model,
            fitness_function=fitness_function,
        )

        init_unit_parameters = np.zeros(shape=(self.n_live_points, model.prior_count))
        init_parameters = np.zeros(shape=(self.n_live_points, model.prior_count))
        init_log_likelihoods = np.zeros(shape=(self.n_live_points))

        for index in range(len(parameters)):

            init_unit_parameters[index, :] = np.asarray(unit_parameters[index])
            init_parameters[index, :] = np.asarray(parameters[index])
            init_log_likelihoods[index] = np.asarray(log_likelihoods[index])

        return [init_unit_parameters, init_parameters, init_log_likelihoods]


class DynestyStatic(AbstractDynesty):
    def __init__(
        self,
        paths=None,
        prior_passer=None,
        n_live_points=None,
        facc=None,
        evidence_tolerance=None,
        bound=None,
        sample=None,
        bootstrap=None,
        enlarge=None,
        update_interval=None,
        vol_dec=None,
        vol_check=None,
        walks=None,
        slices=None,
        fmove=None,
        max_move=None,
        maxiter=None,
        maxcall=None,
        logl_max=None,
        n_effective=None,
        terminate_at_acceptance_ratio=None,
        acceptance_ratio_threshold=None,
        iterations_per_update=None,
        number_of_cores=None,
    ):
        """
        A Dynesty non-linear search using a static number of live points.

        For a full description of Dynesty, checkout its GitHub and readthedocs webpages:

        https://github.com/joshspeagle/dynesty
        https://dynesty.readthedocs.io/en/latest/index.html

        Extensions:

        - Allows runs to be terminated and resumed from the point it was terminated. This is achieved by pickling the
          sampler instance during the model-fit after an input number of iterations.

        Dynesty parameters are also described at the Dynesty API webpage:

        https://dynesty.readthedocs.io/en/latest/api.html

        Parameters
        ----------
        paths : af.Paths
            Manages all paths, e.g. where the search outputs are stored, the samples, backups, etc.
        prior_passer : af.PriorPasser
            Controls how priors are passed from the results of this non-linear search to a subsequent non-linear search.
        facc : float
            The target acceptance fraction for the 'rwalk' sampling option. Default is 0.5. Bounded to be between
            [1. / walks, 1.].
        evidence_threshold : float
            This is called dlogz in the Dynesty API. Iteration will stop when the estimated contribution of the
            remaining prior volume to the total evidence falls below this threshold. Explicitly, the stopping
            criterion is ln(z + z_est) - ln(z) < dlogz, where z is the current evidence from all saved samples and
            z_est is the estimated contribution from the remaining volume. If add_live is True, the default is
            1e-3 * (nlive - 1) + 0.01. Otherwise, the default is 0.01.
        bound : str
            Method used to approximately bound the prior using the current set of live points. Conditions the sampling
            methods used to propose new live points. Choices are no bound ('none'), a single bounding ellipsoid
            ('single'), multiple bounding ellipsoids ('multi'), balls centered on each live point ('balls'), and cubes
            centered on each live point ('cubes'). Default is 'multi'.
        samples : str
            Method used to sample uniformly within the likelihood constraint, conditioned on the provided bounds.
            Unique methods available are: uniform sampling within the bounds('unif'), random walks with fixed
            proposals ('rwalk'), random walks with variable (“staggering”) proposals ('rstagger'), multivariate slice
            sampling along preferred orientations ('slice'), “random” slice sampling along all orientations ('rslice'),
            “Hamiltonian” slices along random trajectories ('hslice'), and any callable function which follows the
            pattern of the sample methods defined in dynesty.sampling. 'auto' selects the sampling method based on the
            dimensionality of the problem (from ndim). When ndim < 10, this defaults to 'unif'. When 10 <= ndim <= 20,
            this defaults to 'rwalk'. When ndim > 20, this defaults to 'hslice' if a gradient is provided and 'slice'
            otherwise. 'rstagger' and 'rslice' are provided as alternatives for 'rwalk' and 'slice', respectively.
            Default is 'auto'.
        bootstrap : int
            Compute this many bootstrapped realizations of the bounding objects. Use the maximum distance found to the
            set of points left out during each iteration to enlarge the resulting volumes. Can lead to unstable
            bounding ellipsoids. Default is 0 (no bootstrap).
        enlarge : float
            Enlarge the volumes of the specified bounding object(s) by this fraction. The preferred method is to
            determine this organically using bootstrapping. If bootstrap > 0, this defaults to 1.0. If bootstrap = 0,
            this instead defaults to 1.25.
        vol_dec : float
            For the 'multi' bounding option, the required fractional reduction in volume after splitting an ellipsoid
            in order to to accept the split. Default is 0.5.
        vol_check : float
            For the 'multi' bounding option, the factor used when checking if the volume of the original bounding
            ellipsoid is large enough to warrant > 2 splits via ell.vol > vol_check * nlive * pointvol. Default is 2.0.
        walks : int
            For the 'rwalk' sampling option, the minimum number of steps (minimum 2) before proposing a new live point.
            Default is 25.
        update_interval : int or float
            If an integer is passed, only update the proposal distribution every update_interval-th likelihood call.
            If a float is passed, update the proposal after every round(update_interval * nlive)-th likelihood call.
            Larger update intervals larger can be more efficient when the likelihood function is quick to evaluate.
            Default behavior is to target a roughly constant change in prior volume, with 1.5 for 'unif', 0.15 * walks
            for 'rwalk' and 'rstagger', 0.9 * ndim * slices for 'slice', 2.0 * slices for 'rslice', and 25.0 * slices
            for 'hslice'.
        slices : int
            For the 'slice', 'rslice', and 'hslice' sampling options, the number of times to execute a “slice update”
            before proposing a new live point. Default is 5. Note that 'slice' cycles through all dimensions when
            executing a “slice update”.
        fmove : float
            The target fraction of samples that are proposed along a trajectory (i.e. not reflecting) for the 'hslice'
            sampling option. Default is 0.9.
        max_move : int
            The maximum number of timesteps allowed for 'hslice' per proposal forwards and backwards in time.
            Default is 100.
        maxiter : int
            Maximum number of iterations. Iteration may stop earlier if the termination condition is reached. Default
            is sys.maxsize (no limit).
        maxcall : int
            Maximum number of likelihood evaluations. Iteration may stop earlier if termination condition is reached.
            Default is sys.maxsize (no limit).
        logl_max : float
            Iteration will stop when the sampled ln(likelihood) exceeds the threshold set by logl_max. Default is no
            bound (np.inf).
        n_effective : int
            Minimum number of effective posterior samples. If the estimated “effective sample size” (ESS) exceeds
            this number, sampling will terminate. Default is no ESS (np.inf).
        terminate_at_acceptance_ratio : bool
            If *True*, the sampler will automatically terminate when the acceptance ratio falls behind an input
            threshold value (see *Nest* for a full description of this feature).
        acceptance_ratio_threshold : float
            The acceptance ratio threshold below which sampling terminates if *terminate_at_acceptance_ratio* is
            *True* (see *Nest* for a full description of this feature).
        iterations_per_update : int
            The number of iterations performed between every Dynesty back-up (via dumping the Dynesty instance as a
            pickle).
        number_of_cores : int
            The number of cores Emcee sampling is performed using a Python multiprocessing Pool instance. If 1, a
            pool instance is not created and the job runs in serial.
        """

        self.n_live_points = (
            self._config("search", "n_live_points", int)
            if n_live_points is None
            else n_live_points
        )

        evidence_tolerance = (
            self._config("search", "evidence_tolerance", float)
            if evidence_tolerance is None
            else evidence_tolerance
        )

        if evidence_tolerance <= 0.0:
            evidence_tolerance = 1e-3 * (self.n_live_points - 1) + 0.01

        super().__init__(
            paths=paths,
            prior_passer=prior_passer,
            n_live_points=n_live_points,
            evidence_tolerance=evidence_tolerance,
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
            maxiter=maxiter,
            maxcall=maxcall,
            logl_max=logl_max,
            n_effective=n_effective,
            iterations_per_update=iterations_per_update,
            terminate_at_acceptance_ratio=terminate_at_acceptance_ratio,
            acceptance_ratio_threshold=acceptance_ratio_threshold,
            number_of_cores=number_of_cores,
        )

        logger.debug("Creating DynestyStatic NLO")

    def sampler_fom_model_and_fitness(self, model, fitness_function):
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
            live_points=live_points,
        )


class DynestyDynamic(AbstractDynesty):
    def __init__(
        self,
        paths=None,
        prior_passer=None,
        n_live_points=None,
        evidence_tolerance=None,
        facc=None,
        bound=None,
        sample=None,
        bootstrap=None,
        enlarge=None,
        update_interval=None,
        vol_dec=None,
        vol_check=None,
        walks=None,
        slices=None,
        fmove=None,
        max_move=None,
        maxiter=None,
        maxcall=None,
        logl_max=None,
        n_effective=None,
        terminate_at_acceptance_ratio=None,
        acceptance_ratio_threshold=None,
        iterations_per_update=None,
        number_of_cores=None,
    ):
        """
        A Dynesty non-linear search, using a dynamically changing number of live points.

        For a full description of Dynesty, checkout its GitHub and readthedocs webpages:

        https://github.com/joshspeagle/dynesty
        https://dynesty.readthedocs.io/en/latest/index.html

        Extensions:

        - Allows runs to be terminated and resumed from the point it was terminated. This is achieved by pickling the
          sampler instance during the model-fit after an input number of iterations.

        Dynesty parameters are also described at the Dynesty API webpage:

        https://dynesty.readthedocs.io/en/latest/api.html

        Parameters
        ----------
        paths : af.Paths
            Manages all paths, e.g. where the search outputs are stored, the samples, backups, etc.
        prior_passer : af.PriorPasser
            Controls how priors are passed from the results of this non-linear search to a subsequent non-linear search.
        facc : float
            The target acceptance fraction for the 'rwalk' sampling option. Default is 0.5. Bounded to be between
            [1. / walks, 1.].
        evidence_threshold : float
            This is called dlogz in the Dynesty API. Iteration will stop when the estimated contribution of the
            remaining prior volume to the total evidence falls below this threshold. Explicitly, the stopping
            criterion is ln(z + z_est) - ln(z) < dlogz, where z is the current evidence from all saved samples and
            z_est is the estimated contribution from the remaining volume. If add_live is True, the default is
            1e-3 * (nlive - 1) + 0.01. Otherwise, the default is 0.01.
        bound : str
            Method used to approximately bound the prior using the current set of live points. Conditions the sampling
            methods used to propose new live points. Choices are no bound ('none'), a single bounding ellipsoid
            ('single'), multiple bounding ellipsoids ('multi'), balls centered on each live point ('balls'), and cubes
            centered on each live point ('cubes'). Default is 'multi'.
        samples : str
            Method used to sample uniformly within the likelihood constraint, conditioned on the provided bounds.
            Unique methods available are: uniform sampling within the bounds('unif'), random walks with fixed
            proposals ('rwalk'), random walks with variable (“staggering”) proposals ('rstagger'), multivariate slice
            sampling along preferred orientations ('slice'), “random” slice sampling along all orientations ('rslice'),
            “Hamiltonian” slices along random trajectories ('hslice'), and any callable function which follows the
            pattern of the sample methods defined in dynesty.sampling. 'auto' selects the sampling method based on the
            dimensionality of the problem (from ndim). When ndim < 10, this defaults to 'unif'. When 10 <= ndim <= 20,
            this defaults to 'rwalk'. When ndim > 20, this defaults to 'hslice' if a gradient is provided and 'slice'
            otherwise. 'rstagger' and 'rslice' are provided as alternatives for 'rwalk' and 'slice', respectively.
            Default is 'auto'.
        bootstrap : int
            Compute this many bootstrapped realizations of the bounding objects. Use the maximum distance found to the
            set of points left out during each iteration to enlarge the resulting volumes. Can lead to unstable
            bounding ellipsoids. Default is 0 (no bootstrap).
        enlarge : float
            Enlarge the volumes of the specified bounding object(s) by this fraction. The preferred method is to
            determine this organically using bootstrapping. If bootstrap > 0, this defaults to 1.0. If bootstrap = 0,
            this instead defaults to 1.25.
        vol_dec : float
            For the 'multi' bounding option, the required fractional reduction in volume after splitting an ellipsoid
            in order to to accept the split. Default is 0.5.
        vol_check : float
            For the 'multi' bounding option, the factor used when checking if the volume of the original bounding
            ellipsoid is large enough to warrant > 2 splits via ell.vol > vol_check * nlive * pointvol. Default is 2.0.
        walks : int
            For the 'rwalk' sampling option, the minimum number of steps (minimum 2) before proposing a new live point.
            Default is 25.
        update_interval : int or float
            If an integer is passed, only update the proposal distribution every update_interval-th likelihood call.
            If a float is passed, update the proposal after every round(update_interval * nlive)-th likelihood call.
            Larger update intervals larger can be more efficient when the likelihood function is quick to evaluate.
            Default behavior is to target a roughly constant change in prior volume, with 1.5 for 'unif', 0.15 * walks
            for 'rwalk' and 'rstagger', 0.9 * ndim * slices for 'slice', 2.0 * slices for 'rslice', and 25.0 * slices
            for 'hslice'.
        slices : int
            For the 'slice', 'rslice', and 'hslice' sampling options, the number of times to execute a “slice update”
            before proposing a new live point. Default is 5. Note that 'slice' cycles through all dimensions when
            executing a “slice update”.
        fmove : float
            The target fraction of samples that are proposed along a trajectory (i.e. not reflecting) for the 'hslice'
            sampling option. Default is 0.9.
        max_move : int
            The maximum number of timesteps allowed for 'hslice' per proposal forwards and backwards in time.
            Default is 100.
        maxiter : int
            Maximum number of iterations. Iteration may stop earlier if the termination condition is reached. Default
            is sys.maxsize (no limit).
        maxcall : int
            Maximum number of likelihood evaluations. Iteration may stop earlier if termination condition is reached.
            Default is sys.maxsize (no limit).
        logl_max : float
            Iteration will stop when the sampled ln(likelihood) exceeds the threshold set by logl_max. Default is no
            bound (np.inf).
        n_effective : int
            Minimum number of effective posterior samples. If the estimated “effective sample size” (ESS) exceeds
            this number, sampling will terminate. Default is no ESS (np.inf).
        terminate_at_acceptance_ratio : bool
            If *True*, the sampler will automatically terminate when the acceptance ratio falls behind an input
            threshold value (see *Nest* for a full description of this feature).
        acceptance_ratio_threshold : float
            The acceptance ratio threshold below which sampling terminates if *terminate_at_acceptance_ratio* is
            *True* (see *Nest* for a full description of this feature).
        iterations_per_update : int
            The number of iterations performed between every Dynesty back-up (via dumping the Dynesty instance as a
            pickle).
        number_of_cores : int
            The number of cores Emcee sampling is performed using a Python multiprocessing Pool instance. If 1, a
            pool instance is not created and the job runs in serial.
        """

        n_live_points = (
            self._config("search", "n_live_points", int)
            if n_live_points is None
            else n_live_points
        )

        if n_live_points <= 0:
            n_live_points = 500

        evidence_tolerance = (
            self._config("search", "evidence_tolerance", float)
            if evidence_tolerance is None
            else evidence_tolerance
        )

        super().__init__(
            paths=paths,
            prior_passer=prior_passer,
            n_live_points=n_live_points,
            evidence_tolerance=evidence_tolerance,
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
            maxiter=maxiter,
            maxcall=maxcall,
            logl_max=logl_max,
            n_effective=n_effective,
            terminate_at_acceptance_ratio=terminate_at_acceptance_ratio,
            acceptance_ratio_threshold=acceptance_ratio_threshold,
            iterations_per_update=iterations_per_update,
            number_of_cores=number_of_cores,
        )

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
        """Perform an update of the non-linear search results, which occurs every *iterations_per_update* of the
        non-linear search. The update performs the following tasks:

        1) Visualize the maximum log likelihood model.
        2) Backup the samples.
        3) Output the model results to the model.reults file.

        These task are performed every n updates, set by the relevent *task_every_update* variable, for example
        *visualize_every_update* and *backup_every_update*.

        Parameters
        ----------
        model : ModelMapper
            The model which generates instances for different points in parameter space.
        analysis : Analysis
            Contains the data and the log likelihood function which fits an instance of the model to the data, returning
            the log likelihood the non-linear search maximizes.
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

        self.save_model_info(model=model)
        self.save_parameter_names_file(model=model)
        self.save_metadata()
        self.save_info(info=info)
        self.save_search()
        self.save_model(model=model)
        # TODO : Better way to handle?
        self.timer.paths = self.paths
        self.timer.start()

        samples = self._fit(model=model, analysis=analysis)
        open(self.paths.has_completed_path, "w+").close()

        return Result(samples=samples, previous_model=model, search=self)

    def _fit(self, model: AbstractPriorModel, analysis) -> Result:
        """
        Fit a model using Dynesty and the Analysis class which contains the data and returns the log likelihood from
        instances of the model, which the non-linear search seeks to maximize.

        Parameters
        ----------
        model : ModelMapper
            The model which generates instances for different points in parameter space.
        analysis : Analysis
            Contains the data and the log likelihood function which fits an instance of the model to the data, returning
            the log likelihood the non-linear search maximizes.

        Returns
        -------
        A result object comprising the Samples object that includes the maximum log likelihood instance and full
        set of accepted ssamples of the fit.
        """

        pool, pool_ids = self.make_pool()

        fitness_function = self.fitness_function_from_model_and_analysis(
            model=model, analysis=analysis
        )

        try:
            os.makedirs(self.paths.samples_path)
        except FileExistsError:
            pass

        sampler = self.sampler_fom_model_and_fitness(
            model=model, fitness_function=fitness_function
        )

        logger.info("No DynestyDynamic samples found, beginning new non-linear search. ")

        # These hacks are necessary to be able to pickle the sampler.

        sampler.rstate = np.random
        sampler.pool = pool

        if self.number_of_cores == 1:
            sampler.M = map
        else:
            sampler.M = pool.map

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
                    nlive_init=self.n_live_points,
                    maxcall=iterations,
                    dlogz_init=self.evidence_tolerance,
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

        if self.should_backup() or not during_analysis:
            self.paths.backup()

        self.timer.update()

        samples = self.samples_from_model(model=model, sampler=sampler)
        samples.write_table(filename=f"{self.paths.sym_path}/samples.csv")
        self.save_samples(samples=samples)

        instance = samples.max_log_likelihood_instance

        if self.should_visualize() or not during_analysis:
            analysis.visualize(instance=instance, during_analysis=during_analysis)

        if self.should_output_model_results() or not during_analysis:

            samples_text.results_to_file(
                samples=samples,
                filename=self.paths.file_results,
                during_analysis=during_analysis,
            )

            samples_text.search_summary_to_file(samples=samples, filename=self.paths.file_search_summary)

        self.paths.backup_zip_remove()

        return samples

    def samples_from_model(self, model, sampler):
        """Create a *Samples* object from this non-linear search's output files on the hard-disk and model.

        For Dynesty, all information that we need is available from the instance of the dynesty sampler.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space. This maps the points from unit
            cube values to physical values via the priors.
        paths : af.Paths
            Manages all paths, e.g. where the search outputs are stored, the samples, backups, etc.
        """

        parameters = sampler.results.samples.tolist()
        log_priors = [
            sum(model.log_priors_from_vector(vector=vector)) for vector in parameters
        ]
        log_likelihoods = list(sampler.results.logl)

        try:
            weights = list(np.exp(np.asarray(sampler.results.logwt) - sampler.results.logz[-1]))
        except:
            weights = sampler.results['weights']

        total_samples = int(np.sum(sampler.results.ncall))
        log_evidence = np.max(sampler.results.logz)

        return NestSamples(
            model=model,
            parameters=parameters,
            log_likelihoods=log_likelihoods,
            log_priors=log_priors,
            weights=weights,
            total_samples=total_samples,
            log_evidence=log_evidence,
            number_live_points=self.n_live_points,
            time=self.timer.time
        )