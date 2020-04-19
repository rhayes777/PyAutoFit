import logging
import math
import os

import emcee
import numpy as np

from autofit import exc
from autofit.plot import samples_text
from autofit.optimize.non_linear import samples
from autofit.optimize.non_linear.non_linear import NonLinearOptimizer
from autofit.optimize.non_linear.non_linear import Result
from autofit.optimize.non_linear.paths import Paths

logger = logging.getLogger(__name__)


class Emcee(NonLinearOptimizer):
    def __init__(self, paths=None, sigma=3):
        """
        Class to setup and run an Emcee non-linear search.

        For a full description of Emcee, checkout its Github and readthedocs webpages:

        https://github.com/dfm/emcee

        https://emcee.readthedocs.io/en/stable/

        **PyAutoFit** extends **emcee** by providing an option to check the auto-correlation length of the chains
        during the run and terminating sampling early if these meet a specified threshold. See this page
        (https://emcee.readthedocs.io/en/stable/tutorials/autocorr/#autocorr) for a description of how this is implemented.

        If you use *emcee* as part of a published work, please cite the package following the instructions under the
        *Attribution* section of the GitHub page.

        Parameters
        ----------
        paths : af.Paths
            A class that manages all paths, e.g. where the phase outputs are stored, the non-linear search chains,
            backups, etc.
        sigma : float
            The error-bound value that linked Gaussian prior withs are computed using. For example, if sigma=3.0,
            parameters will use Gaussian Priors with widths coresponding to errors estimated at 3 sigma confidence.

        Attributes
        ----------
        sigma : float
            The error-bound value that linked Gaussian prior withs are computed using. For example, if sigma=3.0,
            parameters will use Gaussian Priors with widths coresponding to errors estimated at 3 sigma confidence.
        check_auto_correlation : bool
            Whether the auto-correlation lengths of the MCMC chains should be checked to determine the stopping
            criteria. If *True*, this option may terminate the Emcee run before the input number of steps, nsteps, has
            been performed. If *False* nstep samples will be taken.
        auto_correlation_check_size : int
            The length of the chains used to check the auto-correlation lengths (from the latest sample backwards). For
            convergence, the auto-correlations must not change over a certain range of samples. A longer check-size
            thus requires more samples to meet the auto-correlation threshold, taking longer to terminate sampling.
            However, shorter chains risk stopping sampling early due to noise.
        auto_correlation_required_length : int
            The length an auto_correlation chain must be for it to be evaluated whether its change threshold is
            sufficiently small to terminate sampling early.
        auto_correlation_change_threshold : float
            The threshold value by which if the change in auto_correlations is below sampling will be terminated early,
            as it has been determined as converged.

        All remaining attributes are emcee parameters and described at the emcee API webpage:

        https://emcee.readthedocs.io/en/stable/

        """

        if paths is None:
            paths = Paths(non_linear_name=type(self).__name__.lower())

        super().__init__(paths)

        self.sigma = sigma

        self.nwalkers = self.config("nwalkers", int)
        self.nsteps = self.config("nsteps", int)
        self.check_auto_correlation = self.config("check_auto_correlation", bool)
        self.auto_correlation_check_size = self.config(
            "auto_correlation_check_size", int
        )
        self.auto_correlation_required_length = self.config(
            "auto_correlation_required_length", int
        )
        self.auto_correlation_change_threshold = self.config(
            "auto_correlation_change_threshold", float
        )

        logger.debug("Creating Emcee NLO")

    def _simple_fit(self, model, fitness_function):
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
        raise NotImplementedError()

    def copy_with_name_extension(self, extension, remove_phase_tag=False):
        """Copy this instance of the emcee non-linear search with all associated attributes.

        This is used to set up the non-linear search on phase extensions."""
        copy = super().copy_with_name_extension(
            extension=extension, remove_phase_tag=remove_phase_tag
        )
        copy.sigma = self.sigma
        copy.nwalkers = self.nwalkers
        copy.nsteps = self.nsteps
        copy.check_auto_correlation = self.check_auto_correlation
        copy.auto_correlation_check_size = self.auto_correlation_check_size
        copy.auto_correlation_required_length = self.auto_correlation_required_length
        copy.auto_correlation_change_threshold = self.auto_correlation_change_threshold

        return copy

    class Fitness(NonLinearOptimizer.Fitness):
        def __init__(self, paths, analysis, model, samples_from_model):
            super().__init__(paths=paths, analysis=analysis, model=model, samples_from_model=samples_from_model)

            self.accepted_samples = 0

        def fit_instance(self, instance):
            log_likelihood = self.analysis.fit(instance)

            if log_likelihood > self.max_likelihood:

                self.max_likelihood = log_likelihood
                self.result = Result(instance, log_likelihood)

                if self.should_visualize():
                    self.analysis.visualize(instance, during_analysis=True)

                if self.should_backup():
                    self.paths.backup()

                if self.should_output_model_results():

                    samples_text.output_results(samples=self.samples, during_analysis=True)

            return log_likelihood

        def __call__(self, params):

            try:

                instance = self.model.instance_from_vector(vector=params)
                log_likelihood = self.fit_instance(instance)
                log_priors = self.model.log_priors_from_vector(vector=params)

            except exc.FitException:

                return -np.inf

            return log_likelihood + sum(log_priors)

    def _fit(self, analysis, model):

        fitness_function = Emcee.Fitness(
            paths=self.paths,
            analysis=analysis,
            instance_from_vector=model.instance_from_vector,
            log_priors_from_vector=model.log_priors_from_vector,
        )

        emcee_sampler = emcee.EnsembleSampler(
            nwalkers=self.nwalkers,
            ndim=model.prior_count,
            log_prob_fn=fitness_function.__call__,
            backend=emcee.backends.HDFBackend(filename=self.paths.path + "/emcee.hdf"),
        )

        try:
            emcee_state = emcee_sampler.get_last_sample()

            samples = self.samples_from_model(model=model, paths=self.paths)

            previous_run_converged = samples.converged

        except AttributeError:

            emcee_state = np.zeros(shape=(emcee_sampler.nwalkers, emcee_sampler.ndim))

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

                samples = self.samples_from_model(model=model, paths=self.paths)

                if samples.converged and self.check_auto_correlation:
                    break

        logger.info("Emcee complete")

        # TODO: Some of the results below use the backup_path, which isnt updated until the end if thiss function is
        # TODO: not located here. Do we need to rely just ono the optimizer foldeR? This is a good idea if we always
        # TODO: have a valid sym-link( e.g. even for aggregator use).

        self.paths.backup()

        samples = self.samples_from_model(model=model, paths=self.paths)

        instance = samples.max_log_likelihood_instance

        analysis.visualize(instance=instance, during_analysis=False)
        samples.output_results(during_analysis=False)
        samples.output_pdf_plots()
        result = Result(
            instance=instance,
            log_likelihood=samples.max_log_posterior,
            samples=samples,
            previous_model=model,
            gaussian_tuples=samples.gaussian_priors_at_sigma(self.sigma),
        )
        self.paths.backup_zip_remove()
        return result

    def samples_from_model(self, model, paths):
        """Create this non-linear search's output class from the model and paths.

        This function is required by the aggregator, so it knows which output class to generate an instance of."""
        return samples.EmceeSamples(model=model, paths=paths)


