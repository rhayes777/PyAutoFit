import logging
import os
import emcee
import numpy as np
import multiprocessing as mp

from autofit import exc
from autofit.text import samples_text
from autofit.optimize.non_linear import samples
from autofit.optimize.non_linear.non_linear import NonLinearOptimizer
from autofit.optimize.non_linear.non_linear import Result
from autofit.optimize.non_linear.paths import Paths

logger = logging.getLogger(__name__)


class Emcee(NonLinearOptimizer):
    def __init__(
        self,
        paths=None,
        sigma=3,
        nwalkers=None,
        nsteps=None,
        initialize_method=None,
        initialize_ball_lower_limit=None,
        initialize_ball_upper_limit=None,
        auto_correlation_check_for_convergence=None,
        auto_correlation_check_size=None,
        auto_correlation_required_length=None,
        auto_correlation_change_threshold=None,
        number_of_cores=None,
    ):
        """
        Class to setup and run an Emcee non-linear search.

        For a full description of Emcee, checkout its Github and readthedocs webpages:

        https://github.com/dfm/emcee

        https://emcee.readthedocs.io/en/stable/

        Extensions:

        **PyAutoFit** provides the option to check the auto-correlation length of the samples during the run and
        terminating sampling early if these meet a specified threshold. See this page
        (https://emcee.readthedocs.io/en/stable/tutorials/autocorr/#autocorr) for a description.

        **PyAutoFit** also provides different options for walker initialization, with the default 'ball' method
        starting all walkers close to one another in parameter space, as recommended in the Emcee documentation
        (https://emcee.readthedocs.io/en/stable/user/faq/).

        If you use *Emcee* as part of a published work, please cite the package following the instructions under the
        *Attribution* section of the GitHub page.

        Parameters
        ----------
        paths : af.Paths
            A class that manages all paths, e.g. where the phase outputs are stored, the non-linear search samples,
            backups, etc.
        sigma : float
            The error-bound value that linked Gaussian prior withs are computed using. For example, if sigma=3.0,
            parameters will use Gaussian Priors with widths coresponding to errors estimated at 3 sigma confidence.

        Attributes
        ----------
        sigma : float
            The error-bound value that linked Gaussian prior withs are computed using. For example, if sigma=3.0,
            parameters will use Gaussian Priors with widths coresponding to errors estimated at 3 sigma confidence.
        auto_correlation_check_for_convergence : bool
            Whether the auto-correlation lengths of the MCMC samples should be checked to determine the stopping
            criteria. If *True*, this option may terminate the Emcee run before the input number of steps, nsteps, has
            been performed. If *False* nstep samples will be taken.
        initialize_method : str
            The method used to generate where walkers are initialized in parameter space, with options:
            ball (default):
                Walkers are initialized by randomly drawing unit values from a uniform distribution between the
                initialize_ball_lower_limit and initialize_ball_upper_limit values. It is recommended these limits are
                small, such that all walkers begin close to one another.
            prior:
                Walkers are initialized by randomly drawing unit values from a uniform distribution between 0 and 1,
                thus being fully distributed over the prior.
        initialize_ball_upper_limit : float
            The lower limit of the uniform distribution unit values are drawn from when initializing walkers using the
            ball method.
            The upper limit of the uniform distribution unit values are drawn from when initializing walkers using the
            ball method.
        auto_correlation_check_size : int
            The length of the samples used to check the auto-correlation lengths (from the latest sample backwards). For
            convergence, the auto-correlations must not change over a certain range of samples. A longer check-size
            thus requires more samples to meet the auto-correlation threshold, taking longer to terminate sampling.
            However, shorter chains risk stopping sampling early due to noise.
        auto_correlation_required_length : int
            The length an auto_correlation chain must be for it to be evaluated whether its change threshold is
            sufficiently small to terminate sampling early.
        auto_correlation_change_threshold : float
            The threshold value by which if the change in auto_correlations is below sampling will be terminated early,
            as it has been determined as converged.
        number_of_cores : int
            The number of cores Emcee sampling is performed using a Python multiprocessing Pool instance. If 1, a
            pool instance is not created and the job runs in serial.

        All remaining attributes are emcee parameters and described at the emcee API webpage:

        https://emcee.readthedocs.io/en/stable/

        Attributes
        ----------
        sigma : float
            The error-bound value that linked Gaussian prior withs are computed using. For example, if sigma=3.0,
            parameters will use Gaussian Priors with widths coresponding to errors estimated at 3 sigma confidence.

        """

        if paths is None:
            paths = Paths(non_linear_name=type(self).__name__.lower())

        super().__init__(paths=paths)

        self.sigma = sigma

        self.nwalkers = self.config("search", "nwalkers", int) if nwalkers is None else nwalkers
        self.nsteps = self.config("search", "nsteps", int) if nsteps is None else nsteps

        self.initialize_method = (
            self.config("initialize", "method", str)
            if initialize_method is None
            else initialize_method
        )
        self.initialize_ball_lower_limit = (
            self.config("initialize", "ball_lower_limit", float)
            if initialize_ball_lower_limit is None
            else initialize_ball_lower_limit
        )
        self.initialize_ball_upper_limit = (
            self.config("initialize", "ball_upper_limit", float)
            if initialize_ball_upper_limit is None
            else initialize_ball_upper_limit
        )

        self.auto_correlation_check_for_convergence = (
            self.config("auto_correlation", "check_for_convergence", bool)
            if auto_correlation_check_for_convergence is None
            else auto_correlation_check_for_convergence
        )
        self.auto_correlation_check_size = (
            self.config("auto_correlation", "check_size", int)
            if auto_correlation_check_size is None
            else auto_correlation_check_size
        )
        self.auto_correlation_required_length = (
            self.config("auto_correlation", "required_length", int)
            if auto_correlation_required_length is None
            else auto_correlation_required_length
        )
        self.auto_correlation_change_threshold = (
            self.config("auto_correlation", "change_threshold", float)
            if auto_correlation_change_threshold is None
            else auto_correlation_change_threshold
        )

        self.number_of_cores = (
            self.config("parallel", "number_of_cores", int)
            if number_of_cores is None
            else number_of_cores
        )

        logger.debug("Creating Emcee NLO")

    def _fit(self, model, analysis):
        """
        Fit a model using emcee and a function that returns a log likelihood from instances of that model.

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
        of the full chains used by the fit.
        """
        return self._full_fit(model=model, analysis=analysis)

    def copy_with_name_extension(self, extension, remove_phase_tag=False):
        """Copy this instance of the emcee non-linear search with all associated attributes.

        This is used to set up the non-linear search on phase extensions."""
        copy = super().copy_with_name_extension(
            extension=extension, remove_phase_tag=remove_phase_tag
        )
        copy.sigma = self.sigma
        copy.nwalkers = self.nwalkers
        copy.nsteps = self.nsteps
        copy.initialize_method = self.initialize_method
        copy.initialize_ball_lower_limit = self.initialize_ball_lower_limit
        copy.initialize_ball_upper_limit = self.initialize_ball_upper_limit
        copy.auto_correlation_check_for_convergence = self.auto_correlation_check_for_convergence
        copy.auto_correlation_check_size = self.auto_correlation_check_size
        copy.auto_correlation_required_length = self.auto_correlation_required_length
        copy.auto_correlation_change_threshold = self.auto_correlation_change_threshold
        copy.number_of_cores = self.number_of_cores

        return copy

    class Fitness(NonLinearOptimizer.Fitness):
        def __call__(self, params):

            try:

                instance = self.model.instance_from_vector(vector=params)
                log_likelihood = self.fit_instance(instance)
                log_priors = self.model.log_priors_from_vector(vector=params)

            except exc.FitException:

                return -np.inf

            return log_likelihood + sum(log_priors)

    def _full_fit(self, model, analysis):

        pool, pool_ids = self.make_pool()

        fitness_function = self.fitness_function_from_model_and_analysis(
            model=model, analysis=analysis, pool_ids=pool_ids,
        )

        emcee_sampler = emcee.EnsembleSampler(
            nwalkers=self.nwalkers,
            ndim=model.prior_count,
            log_prob_fn=fitness_function.__call__,
            backend=emcee.backends.HDFBackend(filename=self.paths.path + "/emcee.hdf"),
            pool=pool,
        )

        try:

            emcee_state = emcee_sampler.get_last_sample()

            samples = self.samples_from_model(model=model)

            previous_run_converged = samples.converged

        except AttributeError:

            emcee_state = np.zeros(shape=(emcee_sampler.nwalkers, emcee_sampler.ndim))

            if self.initialize_method in "ball":

                for walker_index in range(emcee_sampler.nwalkers):
                    emcee_state[walker_index, :] = np.asarray(
                        model.random_vector_from_priors_within_limits(
                            lower_limit=self.initialize_ball_lower_limit,
                            upper_limit=self.initialize_ball_upper_limit
                        )
                    )

            elif self.initialize_method in "prior":

                for walker_index in range(emcee_sampler.nwalkers):
                    emcee_state[walker_index, :] = np.asarray(
                        model.random_vector_from_priors
                    )

            previous_run_converged = False

        logger.info("Running Emcee Sampling...")

        if self.nsteps - emcee_sampler.iteration > 0 and not previous_run_converged:

            for sample in emcee_sampler.sample(
                initial_state=emcee_state,
                iterations=self.nsteps - emcee_sampler.iteration,
                progress=True,
                skip_initial_state_check=True,
                store=True,
            ):

                if emcee_sampler.iteration % self.auto_correlation_check_size:
                    continue

                samples = self.samples_from_model(model=model)

                if samples.converged and self.auto_correlation_check_for_convergence:
                    break

        logger.info("Emcee complete")

        self.paths.backup()

        samples = self.samples_from_model(model=model)

        analysis.visualize(
            instance=samples.max_log_likelihood_instance, during_analysis=False
        )

        samples_text.results_to_file(
            samples=samples, file_results=self.paths.file_results, during_analysis=False
        )

        self.paths.backup_zip_remove()

        return Result(samples=samples, previous_model=model)

    @property
    def backend(self) -> emcee.backends.HDFBackend:
        """The *Emcee* hdf5 backend, which provides access to all samples, likelihoods, etc. of the non-linear search.

        The sampler is described in the "Results" section at https://dynesty.readthedocs.io/en/latest/quickstart.html"""
        if os.path.isfile(self.paths.sym_path + "/emcee.hdf"):
            return emcee.backends.HDFBackend(
                filename=self.paths.sym_path + "/emcee.hdf"
            )
        else:
            raise FileNotFoundError(
                "The file emcee.hdf does not exist at the path " + self.paths.path
            )

    def fitness_function_from_model_and_analysis(self, model, analysis, pool_ids=None):

        return Emcee.Fitness(
            paths=self.paths,
            model=model,
            analysis=analysis,
            samples_from_model=self.samples_from_model,
            pool_ids=pool_ids,
        )

    def samples_from_model(self, model):
        """Create a *Samples* object from this non-linear search's output files on the hard-disk and model.

        For Emcee, all quantities are extracted via the hdf5 backend of results.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space. This maps the points from unit
            cube values to physical values via the priors.
        paths : af.Paths
            A class that manages all paths, e.g. where the phase outputs are stored, the non-linear search chains,
            backups, etc.
        """

        parameters = self.backend.get_chain(flat=True)
        log_priors = [
            sum(model.log_priors_from_vector(vector=vector)) for vector in parameters
        ]
        log_likelihoods = self.backend.get_log_prob(flat=True)
        weights = len(log_likelihoods) * [1.0]
        auto_correlation_time = self.backend.get_autocorr_time(tol=0)
        total_walkers = len(self.backend.get_chain()[0, :, 0])
        total_steps = len(self.backend.get_log_prob())

        return samples.MCMCSamples(
            model=model,
            parameters=parameters,
            log_likelihoods=log_likelihoods,
            log_priors=log_priors,
            weights=weights,
            total_walkers=total_walkers,
            total_steps=total_steps,
            auto_correlation_times=auto_correlation_time,
            auto_correlation_check_size=self.auto_correlation_check_size,
            auto_correlation_required_length=self.auto_correlation_required_length,
            auto_correlation_change_threshold=self.auto_correlation_change_threshold,
            backend=self.backend,
        )
