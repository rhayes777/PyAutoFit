import logging
import math
import os

import emcee
import numpy as np

from autofit import conf, exc
from autofit.optimize.non_linear.non_linear import NonLinearOptimizer
from autofit.optimize.non_linear.non_linear import Result
from autofit.optimize.non_linear.output import MCMCOutput
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
        Fit a model using emcee and a function that returns a likelihood from instances of that model.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space. This maps the points from unit
            cube values to physical values via the priors.
        fitness_function
            A function that fits this model to the data, returning the likelihood of the fit.

        Returns
        -------
        A result object comprising the best-fit model instance, likelihood and an *Output* class that enables analysis
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
        def __init__(self, paths, analysis, instance_from_vector, log_priors_from_vector, output_results):
            super().__init__(paths, analysis, output_results)
            self.instance_from_vector = instance_from_vector
            self.log_priors_from_vector = log_priors_from_vector
            self.accepted_samples = 0

        def fit_instance(self, instance):
            likelihood = self.analysis.fit(instance)

            if likelihood > self.max_likelihood:

                self.max_likelihood = likelihood
                self.result = Result(instance, likelihood)

                if self.should_visualize():
                    self.analysis.visualize(instance, during_analysis=True)

                if self.should_backup():
                    self.paths.backup()

                if self.should_output_model_results():
                    self.output_results(during_analysis=True)

            return likelihood

        def __call__(self, params):

            try:

                instance = self.instance_from_vector(params)
                log_likelihood = self.fit_instance(instance)
                log_priors = self.log_priors_from_vector(params)

            except exc.FitException:

                return -np.inf

            return log_likelihood + sum(log_priors)

    def _fit(self, analysis, model):

        output = EmceeOutput(
            model=model,
            paths=self.paths,
            auto_correlation_check_size=self.auto_correlation_check_size,
            auto_correlation_required_length=self.auto_correlation_required_length,
            auto_correlation_change_threshold=self.auto_correlation_change_threshold,
        )

        fitness_function = Emcee.Fitness(
            paths=self.paths,
            analysis=analysis,
            instance_from_vector=model.instance_from_vector,
            log_priors_from_vector=model.log_priors_from_vector,
            output_results=output.output_results,
        )

        emcee_sampler = emcee.EnsembleSampler(
            nwalkers=self.nwalkers,
            ndim=model.prior_count,
            log_prob_fn=fitness_function.__call__,
            backend=emcee.backends.HDFBackend(filename=self.paths.path + "/emcee.hdf"),
        )

        output.save_model_info()

        try:
            emcee_state = emcee_sampler.get_last_sample()
            previous_run_converged = output.converged

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

                if output.converged and self.check_auto_correlation:
                    break

        logger.info("Emcee complete")

        # TODO: Some of the results below use the backup_path, which isnt updated until the end if thiss function is
        # TODO: not located here. Do we need to rely just ono the optimizer foldeR? This is a good idea if we always
        # TODO: have a valid sym-link( e.g. even for aggregator use).

        self.paths.backup()

        instance = output.most_likely_instance

        analysis.visualize(instance=instance, during_analysis=False)
        output.output_results(during_analysis=False)
        output.output_pdf_plots()
        result = Result(
            instance=instance,
            likelihood=output.maximum_log_likelihood,
            output=output,
            previous_model=model,
            gaussian_tuples=output.gaussian_priors_at_sigma(self.sigma),
        )
        self.paths.backup_zip_remove()
        return result

    def output_from_model(self, model, paths):
        """Create this non-linear search's output class from the model and paths.

        This function is required by the aggregator, so it knows which output class to generate an instance of."""
        return EmceeOutput(model=model, paths=paths)


class EmceeOutput(MCMCOutput):
    def __init__(
        self,
        model,
        paths,
        auto_correlation_check_size,
        auto_correlation_required_length,
        auto_correlation_change_threshold,
    ):

        super(EmceeOutput, self).__init__(model=model, paths=paths)

        self.auto_correlation_check_size = auto_correlation_check_size
        self.auto_correlation_required_length = auto_correlation_required_length
        self.auto_correlation_change_threshold = auto_correlation_change_threshold

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

    @property
    def pdf(self):
        """An interface to *GetDist* which can be used for analysing and visualizing the non-linear search chains.

        *GetDist* can only be used when chains are converged enough to provide a smooth PDF and this convergence is
        checked using the *pdf_converged* bool before *GetDist* is called.

        https://github.com/cmbant/getdist
        https://getdist.readthedocs.io/en/latest/

        For *emcee*, chains are passed to *GetDist* via the hdt backend. *GetDist* currently does not provide accurate
        model sampling.
        """
        import getdist

        return getdist.mcsamples.MCSamples(samples=self.samples_after_burn_in)

    @property
    def pdf_converged(self):
        """ To analyse and visualize chains using *GetDist*, the analysis must be sufficiently converged to produce
        smooth enough PDF for analysis. This property checks whether the non-linear search's chains are sufficiently
        converged for *GetDist* use.

        Emcee chains can be analysed by GetDist irrespective of how long the sampler has run, albeit low run times
        will likely produce inaccurate results."""
        return True

    @property
    def total_samples(self) -> int:
        """The total number of samples performed by the non-linear search.

        For Emcee, this includes all accepted and rejected proposed steps and is loaded from the results backend.
        """
        return len(self.backend.get_chain(flat=True))

    @property
    def total_walkers(self) -> int:
        """The total number of walkers used by this *Emcee* non-linear search.
        """
        return len(self.backend.get_chain()[0, :, 0])

    @property
    def total_steps(self) -> int:
        """The total number of steps taken by each walk of this *Emcee* non-linear search.
        """
        return len(self.backend.get_log_prob())

    @property
    def maximum_log_likelihood(self) -> float:
        """The maximum log likelihood value of the non-linear search, corresponding to the best-fit model.

        For emcee, this is computed from the backend's list of all likelihood values."""
        return self.backend.get_log_prob(flat=True)[self.most_likely_index]

    @property
    def samples_after_burn_in(self) -> [list]:
        """The emcee samples with the initial burn-in samples removed.

        The burn-in period is estimated using the auto-correlation times o the parameters."""

        discard = int(3.0 * np.max(self.auto_correlation_times_of_parameters))
        thin = int(np.max(self.auto_correlation_times_of_parameters) / 2.0)
        return self.backend.get_chain(discard=discard, thin=thin, flat=True)

    @property
    def auto_correlation_times_of_parameters(self) -> [float]:
        """Estimate the autocorrelation time of all parameters from the emcee backend results."""
        return self.backend.get_autocorr_time(tol=0)

    @property
    def previous_auto_correlation_times_of_parameters(self) -> [float]:
        return emcee.autocorr.integrated_time(
            x=self.backend.get_chain()[: -self.auto_correlation_check_size, :, :], tol=0
        )

    @property
    def relative_auto_correlation_times(self) -> [float]:
        return (
            np.abs(
                self.previous_auto_correlation_times_of_parameters
                - self.auto_correlation_times_of_parameters
            )
            / self.auto_correlation_times_of_parameters
        )

    @property
    def converged(self) -> bool:
        """Whether the emcee chains have converged on a solution or if they are still in a burn-in period, based on the 
        auto correlation times of parameters."""
        converged = np.all(
            self.auto_correlation_times_of_parameters
            * self.auto_correlation_required_length
            < self.total_samples
        )
        if converged:
            try:
                converged &= np.all(
                    self.relative_auto_correlation_times
                    < self.auto_correlation_change_threshold
                )
            except IndexError:
                return False
        return converged

    @property
    def most_probable_vector(self) -> [float]:
        """ The median of the probability density function (PDF) of every parameter marginalized in 1D, returned
        as a list of values.

        This is computed by binning all sampls after burn-in into a histogram and take its median (e.g. 50%) value. """
        samples = self.samples_after_burn_in
        return [
            float(np.percentile(samples[:, i], [50]))
            for i in range(self.model.prior_count)
        ]

    @property
    def most_likely_index(self) -> int:
        """The index of the accepted sample with the highest likelihood, e.g. that of best-fit / most_likely model."""
        return int(np.argmax(self.backend.get_log_prob(flat=True)))

    @property
    def most_likely_vector(self) -> [float]:
        """ The best-fit model sampled by the non-linear search (corresponding to the maximum log-likelihood), returned
        as a list of values.

        The vector is read from the results backend instance, by first locating the index corresponding to the highest
        likelihood accepted sample."""
        return self.backend.get_chain(flat=True)[self.most_likely_index]

    def vector_at_sigma(self, sigma) -> [float]:
        """ The value of every parameter marginalized in 1D at an input sigma value of its probability density function
        (PDF), returned as two lists of values corresponding to the lower and upper values parameter values.

        For example, if sigma is 1.0, the marginalized values of every parameter at 31.7% and 68.2% percentiles of each
        PDF is returned.

        This does not account for covariance between parameters. For example, if two parameters (x, y) are degenerate
        whereby x decreases as y gets larger to give the same PDF, this function will still return both at their
        upper values. Thus, caution is advised when using the function to reperform a model-fits.

        For *Emcee*, if the chains have converged this is estimated by binning the samples after burn-in into a
        histogram and taking the parameter values at the input PDF %.

        Parameters
        ----------
        sigma : float
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF)."""
        limit = math.erf(0.5 * sigma * math.sqrt(2))

        samples = self.samples_after_burn_in

        return [
            tuple(np.percentile(samples[:, i], [100.0 * (1.0 - limit), 100.0 * limit]))
            for i in range(self.model.prior_count)
        ]

    def vector_from_sample_index(self, sample_index) -> [float]:
        """The model parameters of an individual sample of the non-linear search.

        Parameters
        ----------
        sample_index : int
            The index of the sample in the non-linear search, e.g. 0 gives the first sample.
        """
        return list(self.pdf.samples[sample_index])

    def weight_from_sample_index(self, sample_index) -> [float]:
        """The weight of an individual sample of the non-linear search.

        Parameters
        ----------
        sample_index : int
            The index of the sample in the non-linear search, e.g. 0 gives the first sample.
        """
        return self.pdf.weights[sample_index]

    def likelihood_from_sample_index(self, sample_index) -> [float]:
        """The likelihood of an individual sample of the non-linear search.

        Parameters
        ----------
        sample_index : int
            The index of the sample in the non-linear search, e.g. 0 gives the first sample.
        """
        return self.backend.get_log_prob(flat=True)[sample_index]

    def output_pdf_plots(self):

        pass
