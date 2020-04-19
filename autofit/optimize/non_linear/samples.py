import numpy as np
import logging
import math

from autofit import conf
from autofit.mapper import model
from autofit.tools import text_formatter, text_util

logger = logging.getLogger(__name__)


class AbstractSamples:
    def __init__(self, model, parameters, log_likelihoods, log_priors, weights, unconverged_sample_size=100):
        """The *Output* classes in **PyAutoFit** provide an interface between the results of a non-linear search (e.g.
        as files on your hard-disk) and Python.

        For example, the output class can be used to load an instance of the best-fit model, get an instance of any
        individual sample by the non-linear search and return information on the likelihoods, errors, etc.

        Parameters
        ----------
        model : af.ModelMapper
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        """
        self.model = model
        self.parameters = parameters
        self.log_likelihoods = log_likelihoods
        self.log_priors = log_priors
        self.weights = weights
        self.log_posteriors = list(map(lambda lh, prior : lh * prior, log_likelihoods, log_priors))
        self._unconverged_sample_size = unconverged_sample_size

    @property
    def total_samples(self) -> int:
        """The total number of samples performed by the non-linear search."""
        return len(self.log_likelihoods)

    @property
    def unconverged_sample_size(self):
        """If a set of samples are unconverged, alternative methods to compute their means, errors, etc are used as
        an alternative to GetDist.

        These use a subset of samples spanning the range from the most recent sample to the valaue of the
        unconverted_sample_size. However, if there are fewer samples than this size, we change the size to be the
         the size of the total number of samples"""
        if self.total_samples > self._unconverged_sample_size:
            return self._unconverged_sample_size
        return self.total_samples

    @property
    def max_log_likelihood_index(self) -> int:
        """The index of the sample with the highest log likelihood."""
        return int(np.argmax(self.log_likelihoods))

    @property
    def max_log_likelihood_vector(self) -> [float]:
        """ The parameters of the maximum log likelihood sample of the non-linear search returned as a list of values."""
        return self.parameters[self.max_log_likelihood_index]

    @property
    def max_log_likelihood_instance(self) -> model.ModelInstance:
        """  The parameters of the maximum log likelihood sample of the non-linear search returned as a model instance."""
        return self.model.instance_from_vector(vector=self.max_log_likelihood_vector)

    @property
    def max_log_posterior_index(self) -> int:
        """The index of the sample with the highest log posterior."""
        return int(np.argmax(self.log_posteriors))

    @property
    def max_log_posterior_vector(self) -> [float]:
        """ The parameters of the maximum log posterior sample of the non-linear search returned as a list of values."""
        return self.parameters[self.max_log_posterior_index]

    @property
    def max_log_posterior_instance(self) -> model.ModelInstance:
        """  The parameters of the maximum log posterior sample of the non-linear search returned as a model instance."""
        return self.model.instance_from_vector(vector=self.max_log_posterior_vector)

    @property
    def pdf_converged(self) -> bool:
        """ To analyse and visualize chains using *GetDist*, the analysis must be sufficiently converged to produce
        smooth enough PDF for analysis. This property checks whether the non-linear search's chains are sufficiently
        converged for *GetDist* use.

        For *Dynesty*, during initial sampling one accepted live point typically has > 99% of the probabilty as its
        log_likelihood is significantly higher than all other points. Convergence is only achieved late in sampling when
        all live points have similar log_likelihood and sampling probabilities."""
        try:
            densities_1d = list(
                map(lambda p: self.pdf.get1DDensity(p), self.pdf.getParamNames().names)
            )

            if densities_1d == []:
                return False

            return True
        except Exception:
            return False

    @property
    def pdf(self):
        """An interface to *GetDist* which can be used for analysing and visualizing the non-linear search chains.

        *GetDist* can only be used when chains are converged enough to provide a smooth PDF and this convergence is
        checked using the *pdf_converged* bool before *GetDist* is called.

        https://github.com/cmbant/getdist
        https://getdist.readthedocs.io/en/latest/

        For *Dynesty*, chains are passed to *GetDist* using the pickled sapler instance, which contains the physical
        model parameters of every accepted sample, their likelihoods and weights.
        """
        import getdist

        return getdist.mcsamples.MCSamples(
            samples=self.parameters,
            weights=self.weights,
            loglikes=self.log_likelihoods,
        )

    @property
    def most_probable_vector(self) -> [float]:
        """ The median of the probability density function (PDF) of every parameter marginalized in 1D, returned
        as a list of values.

        If the chains are sufficiently converged this is estimated by passing the accepted samples to *GetDist*, else
        a crude estimate using the mean value of all accepted samples is used."""
        if self.pdf_converged:
            return self.pdf.getMeans()
        else:
            return list(np.mean(self.parameters, axis=0))

    @property
    def most_probable_instance(self) -> model.ModelInstance:
        """ The median of the probability density function (PDF) of every parameter marginalized in 1D, returned
        as a model instance."""
        return self.model.instance_from_vector(vector=self.most_probable_vector)

    def vector_at_sigma(self, sigma) -> [float]:
        """ The value of every parameter marginalized in 1D at an input sigma value of its probability density function
        (PDF), returned as two lists of values corresponding to the lower and upper values parameter values.

        For example, if sigma is 1.0, the marginalized values of every parameter at 31.7% and 68.2% percentiles of each
        PDF is returned.

        This does not account for covariance between parameters. For example, if two parameters (x, y) are degenerate
        whereby x decreases as y gets larger to give the same PDF, this function will still return both at their
        upper values. Thus, caution is advised when using the function to reperform a model-fits.

        For *Dynesty*, this is estimated using *GetDist* if the chains have converged, by sampling the density
        function at an input PDF %. If not converged, a crude estimate using the range of values of the current
        physical live points is used.

        Parameters
        ----------
        sigma : float
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF)."""
        limit = math.erf(0.5 * sigma * math.sqrt(2))

        if self.pdf_converged:
            densities_1d = list(
                map(lambda p: self.pdf.get1DDensity(p), self.pdf.getParamNames().names)
            )

            return list(map(lambda p: p.getLimits(limit), densities_1d))

        print(self.parameters[-self.unconverged_sample_size :])

        parameters_min = list(
            np.min(self.parameters[-self.unconverged_sample_size :], axis=0)
        )
        parameters_max = list(
            np.max(self.parameters[-self.unconverged_sample_size :], axis=0)
        )

        return [
            (parameters_min[index], parameters_max[index])
            for index in range(len(parameters_min))
        ]

    def vector_at_upper_sigma(self, sigma) -> [float]:
        """The upper value of every parameter marginalized in 1D at an input sigma value of its probability density
        function (PDF), returned as a list.

        See *vector_at_sigma* for a full description of how the parameters at sigma are computed.

        Parameters
        -----------
        sigma : float
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the \
            PDF).
        """
        return list(map(lambda param: param[1], self.vector_at_sigma(sigma)))

    def vector_at_lower_sigma(self, sigma) -> [float]:
        """The lower value of every parameter marginalized in 1D at an input sigma value of its probability density
        function (PDF), returned as a list.

        See *vector_at_sigma* for a full description of how the parameters at sigma are computed.

        Parameters
        -----------
        sigma : float
            The sigma limit within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the \
            PDF).
        """
        return list(map(lambda param: param[0], self.vector_at_sigma(sigma)))

    def instance_at_sigma(self, sigma) -> model.ModelInstance:
        """ The value of every parameter marginalized in 1D at an input sigma value of its probability density function
        (PDF), returned as a list of model instances corresponding to the lower and upper values estimated from the PDF.

        See *vector_at_sigma* for a full description of how the parameters at sigma are computed.

        Parameters
        ----------
        sigma : float
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF)."""
        return self.model.instance_from_vector(
            vector=self.vector_at_sigma(sigma=sigma), assert_priors_in_limits=False
        )

    def instance_at_upper_sigma(self, sigma) -> model.ModelInstance:
        """The upper value of every parameter marginalized in 1D at an input sigma value of its probability density
        function (PDF), returned as a model instance.

        See *vector_at_sigma* for a full description of how the parameters at sigma are computed.

        Parameters
        -----------
        sigma : float
            The sigma limit within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the \
            PDF).
        """
        return self.model.instance_from_vector(
            vector=self.vector_at_upper_sigma(sigma=sigma),
            assert_priors_in_limits=False,
        )

    def instance_at_lower_sigma(self, sigma) -> model.ModelInstance:
        """The lower value of every parameter marginalized in 1D at an input sigma value of its probability density
        function (PDF), returned as a model instance.

        See *vector_at_sigma* for a full description of how the parameters at sigma are computed.

        Parameters
        -----------
        sigma : float
            The sigma limit within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the \
            PDF).
        """
        return self.model.instance_from_vector(
            vector=self.vector_at_lower_sigma(sigma=sigma),
            assert_priors_in_limits=False,
        )

    def error_vector_at_sigma(self, sigma) -> [float]:
        """ The value of every error after marginalization in 1D at an input sigma value of the probability density
        function (PDF), returned as two lists of values corresponding to the lower and upper errors.

        For example, if sigma is 1.0, the errors marginalized at 31.7% and 68.2% percentiles of each PDF is returned.

        Parameters
        ----------
        sigma : float
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF)."""
        uppers = self.vector_at_upper_sigma(sigma=sigma)
        lowers = self.vector_at_lower_sigma(sigma=sigma)
        return list(map(lambda upper, lower: upper - lower, uppers, lowers))

    def error_vector_at_upper_sigma(self, sigma) -> [float]:
        """The upper error of every parameter marginalized in 1D at an input sigma value of its probability density
        function (PDF), returned as a list.

        See *vector_at_sigma* for a full description of how the parameters at sigma are computed.

        Parameters
        -----------
        sigma : float
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the \
            PDF).
        """
        uppers = self.vector_at_upper_sigma(sigma=sigma)
        return list(
            map(
                lambda upper, most_probable: upper - most_probable,
                uppers,
                self.most_probable_vector,
            )
        )

    def error_vector_at_lower_sigma(self, sigma) -> [float]:
        """The lower error of every parameter marginalized in 1D at an input sigma value of its probability density
        function (PDF), returned as a list.

        See *vector_at_sigma* for a full description of how the parameters at sigma are computed.

        Parameters
        -----------
        sigma : float
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the \
            PDF).
        """
        lowers = self.vector_at_lower_sigma(sigma=sigma)
        return list(
            map(
                lambda lower, most_probable: most_probable - lower,
                lowers,
                self.most_probable_vector,
            )
        )

    def error_instance_at_sigma(self, sigma) -> model.ModelInstance:
        """ The error of every parameter marginalized in 1D at an input sigma value of its probability density function
        (PDF), returned as a list of model instances corresponding to the lower and upper errors.

        See *vector_at_sigma* for a full description of how the parameters at sigma are computed.

        Parameters
        ----------
        sigma : float
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF)."""
        return self.model.instance_from_vector(
            vector=self.error_vector_at_sigma(sigma=sigma)
        )

    def error_instance_at_upper_sigma(self, sigma) -> model.ModelInstance:
        """The upper error of every parameter marginalized in 1D at an input sigma value of its probability density
        function (PDF), returned as a model instance.

        See *vector_at_sigma* for a full description of how the parameters at sigma are computed.

        Parameters
        -----------
        sigma : float
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the \
            PDF).
        """
        return self.model.instance_from_vector(
            vector=self.error_vector_at_upper_sigma(sigma=sigma)
        )

    def error_instance_at_lower_sigma(self, sigma) -> model.ModelInstance:
        """The lower error of every parameter marginalized in 1D at an input sigma value of its probability density
        function (PDF), returned as a model instance.

        See *vector_at_sigma* for a full description of how the parameters at sigma are computed.

        Parameters
        -----------
        sigma : float
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the \
            PDF).
        """
        return self.model.instance_from_vector(
            vector=self.error_vector_at_lower_sigma(sigma=sigma)
        )

    def gaussian_priors_at_sigma(self, sigma) -> [list]:
        """*GaussianPrior*s of every parameter used to link its inferred values and errors to priors used to sample the
        same (or similar) parameters in a subsequent phase, where:

         - The mean is given by their most-probable values (using *most_probable_vector*).
         - Their errors are computed at an input sigma value (using *errors_at_sigma*).

        Parameters
        -----------
        sigma : float
            The sigma limit within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the \
            PDF).
        """

        means = self.most_probable_vector
        uppers = self.vector_at_upper_sigma(sigma=sigma)
        lowers = self.vector_at_lower_sigma(sigma=sigma)

        # noinspection PyArgumentList
        sigmas = list(
            map(
                lambda mean, upper, lower: max([upper - mean, mean - lower]),
                means,
                uppers,
                lowers,
            )
        )

        return list(map(lambda mean, sigma: (mean, sigma), means, sigmas))

    def log_likelihood_from_sample_index(self, sample_index) -> float:
        """The log likelihood of an individual sample of the non-linear search.

        Parameters
        ----------
        sample_index : int
            The index of the sample in the non-linear search, e.g. 0 gives the first sample.
        """
        raise NotImplementedError()

    def vector_from_sample_index(self, sample_index) -> [float]:
        """The parameters of an individual sample of the non-linear search, returned as a 1D list.

        Parameters
        ----------
        sample_index : int
            The index of the sample in the non-linear search, e.g. 0 gives the first sample.
        """
        raise NotImplementedError()

    def instance_from_sample_index(self, sample_index) -> model.ModelInstance:
        """The parameters of an individual saple of the non-linear search, returned as a model instance.

        Parameters
        -----------
        sample_index : int
            The sample index of the weighted sample to return.
        """
        model_parameters = self.vector_from_sample_index(sample_index=sample_index)

        return self.model.instance_from_vector(vector=model_parameters)

    def offset_vector_from_input_vector(self, input_vector) -> [float]:
        """ The values of an input_vector offset by the *most_probable_vector* (the PDF medians).

        If the 'true' values of a model are known and input as the *input_vector*, this function returns the results
        of the non-linear search as values offset from the 'true' model. For example, a value 0.0 means the non-linear
        search estimated the model parameter value correctly.

        Parameters
        ----------

        input_vector : list
            A 1D list of values the most-probable model is offset by. This list must have the same dimensions as free
            parameters in the model-fit.
        """
        return list(
            map(
                lambda input, most_probable: most_probable - input,
                input_vector,
                self.most_probable_vector,
            )
        )

    def output_pdf_plots(self):
        """Output plots of the probability density functions of the non-linear seach."""
        raise NotImplementedError()


class MCMCSamples(AbstractSamples):
    pass


class EmceeSamples(MCMCSamples):
    def __init__(
        self,
        model,
        paths,
        auto_correlation_check_size,
        auto_correlation_required_length,
        auto_correlation_change_threshold,
    ):

        super(EmceeSamples, self).__init__(model=model, paths=paths)

        self.auto_correlation_check_size = auto_correlation_check_size
        self.auto_correlation_required_length = auto_correlation_required_length
        self.auto_correlation_change_threshold = auto_correlation_change_threshold

    # @property
    # def backend(self) -> emcee.backends.HDFBackend:
    #     """The *Emcee* hdf5 backend, which provides access to all samples, likelihoods, etc. of the non-linear search.
    #
    #     The sampler is described in the "Results" section at https://dynesty.readthedocs.io/en/latest/quickstart.html"""
    #     if os.path.isfile(self.paths.sym_path + "/emcee.hdf"):
    #         return emcee.backends.HDFBackend(
    #             filename=self.paths.sym_path + "/emcee.hdf"
    #         )
    #     else:
    #         raise FileNotFoundError(
    #             "The file emcee.hdf does not exist at the path " + self.paths.path
    #         )

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

    # @property
    # def samples_after_burn_in(self) -> [list]:
    #     """The emcee samples with the initial burn-in samples removed.
    #
    #     The burn-in period is estimated using the auto-correlation times o the parameters."""
    #
    #     discard = int(3.0 * np.max(self.auto_correlation_times_of_parameters))
    #     thin = int(np.max(self.auto_correlation_times_of_parameters) / 2.0)
    #     return self.backend.get_chain(discard=discard, thin=thin, flat=True)

    @property
    def auto_correlation_times_of_parameters(self) -> [float]:
        """Estimate the autocorrelation time of all parameters from the emcee backend results."""
        return self.backend.get_autocorr_time(tol=0)

    # @property
    # def previous_auto_correlation_times_of_parameters(self) -> [float]:
    #     return emcee.autocorr.integrated_time(
    #         x=self.backend.get_chain()[: -self.auto_correlation_check_size, :, :], tol=0
    #     )

    # @property
    # def relative_auto_correlation_times(self) -> [float]:
    #     return (
    #         np.abs(
    #             self.previous_auto_correlation_times_of_parameters
    #             - self.auto_correlation_times_of_parameters
    #         )
    #         / self.auto_correlation_times_of_parameters
    #     )

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
    def log_likelihoods(self) -> [float]:
        """A list of log likelihood values of every sample of the Emcee chains.

        The log likelihood is the value sampled via the log likelihood function of a model and does not have the log_prior
        values added to it. This is not directly sampled by Emcee and is thus computed by re-subtracting off all
        log_prior values."""
        params = self.backend.get_chain(flat=True)
        log_priors = [sum(self.model.log_priors_from_vector(vector=vector)) for vector in params]
        log_posteriors = self.backend.get_log_prob(flat=True)

        return list(map(lambda log_prior, log_posterior : log_posterior - log_prior, log_priors, log_posteriors))

    @property
    def max_log_likelihood_index(self) -> int:
        """The index of the accepted sample with the highest log likelihood.

        The log likelihood is the value sampled via the log likelihood function of a model and does not have the log_prior
        values added to it. This is not directly sampled by Emcee and is thus computed by re-subtracting off all
        log_prior values."""
        return int(np.argmax(self.log_likelihoods))

    @property
    def max_log_likelihood(self) -> float:
        """The maximum log likelihood value of the non-linear search, corresponding to the best-fit model.

        The log likelihood is the value sampled via the log likelihood function of a model and does not have the log_prior
        values added to it. This is not directly sampled by Emcee and is thus computed by re-subtracting off all
        log_prior values."""
        return self.log_likelihoods[self.max_log_likelihood_index]

    @property
    def max_log_likelihood_vector(self) -> [float]:
        """ The vector of parameters corresponding to the highest log likelihood sample, returned as a list of
        parameter values.

        The log likelihood is the value sampled via the log likelihood function of a model and does not have the log_prior
        values added to it. This is not directly sampled by Emcee and is thus computed by re-subtracting off all
        log_prior values."""
        return self.backend.get_chain(flat=True)[self.max_log_likelihood_index]

    @property
    def max_log_posterior_index(self) -> int:
        """The index of the accepted sample with the highest posterior value.

        This is directly extracted from the Emcee results backend."""
        return int(np.argmax(self.backend.get_log_prob(flat=True)))

    @property
    def max_log_posterior(self) -> float:
        """The maximum posterior value of a sample in the non-linear search.

        For emcee, this is computed from the backend's list of all posterior values."""
        return self.backend.get_log_prob(flat=True)[self.max_log_posterior_index]

    @property
    def max_log_posterior_vector(self) -> [float]:
        """ The vector of parameters corresponding to the highest log-posterior sample, returned as a list of
        parameter values.

        The vector is read from the Emcee results backend, using the index of the highest posterior sample."""
        return self.backend.get_chain(flat=True)[self.max_log_posterior_index]

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

    def log_likelihood_from_sample_index(self, sample_index) -> [float]:
        """The log likelihood of an individual sample of the non-linear search.

        This is computed by subtracting the log prior from the log posterior.

        Parameters
        ----------
        sample_index : int
            The index of the sample in the non-linear search, e.g. 0 gives the first sample.
        """
        return self.log_posterior_from_sample_index(sample_index=sample_index) - \
               self.log_prior_from_sample_index(sample_index=sample_index)

    def log_prior_from_sample_index(self, sample_index) -> [float]:
        """The sum of log priors of all parameters of an individual sample of the non-linear search.

        This is computed using the physical values of each parameter and their prior.

        Parameters
        ----------
        sample_index : int
            The index of the sample in the non-linear search, e.g. 0 gives the first sample.
        """
        vector = self.vector_from_sample_index(sample_index=sample_index)
        return sum(self.model.log_priors_from_vector(vector=vector))

    def log_posterior_from_sample_index(self, sample_index) -> [float]:
        """The log of the posterior of an individual sample of the non-linear search.

        This is directly extracted from the Emcee samples.

        Parameters
        ----------
        sample_index : int
            The index of the sample in the non-linear search, e.g. 0 gives the first sample.
        """
        return self.backend.get_log_prob(flat=True)[sample_index]

    def output_pdf_plots(self):

        pass


class NestedSamplerSamples(AbstractSamples):
    @property
    def number_live_points(self) -> int:
        """The number of live points used by the nested sampler."""
        raise NotImplementedError()

    @property
    def total_accepted_samples(self) -> int:
        """The total number of accepted samples performed by the non-linear search.
        """
        raise NotImplementedError()

    @property
    def log_evidence(self) -> float:
        """The Bayesian log evidence estimated by the nested sampling algorithm."""
        raise NotImplementedError()

    def weight_from_sample_index(self, sample_index) -> float:
        """The weight of an individual sample of the non-linear search.

        Parameters
        ----------
        sample_index : int
            The index of the sample in the non-linear search, e.g. 0 gives the first sample.
        """
        raise NotImplementedError()

    def output_pdf_plots(self):
        """Output plots of the probability density functions of the non-linear seach.

        This uses *GetDist* to plot:

         - The marginalize 1D PDF of every parameter.
         - The marginalized 2D PDF of every parameter pair.
         - A Triangle plot of the 2D and 1D PDF's.
         """
        import getdist.plots
        import matplotlib

        backend = conf.instance.visualize_general.get("general", "backend", str)
        if not backend in "default":
            matplotlib.use(backend)
        import matplotlib.pyplot as plt

        pdf_plot = getdist.plots.GetDistPlotter()

        plot_pdf_1d_params = conf.instance.visualize_plots.get("pdf", "1d_params", bool)

        if plot_pdf_1d_params:

            for param_name in self.model.param_names:
                pdf_plot.plot_1d(roots=self.pdf, param=param_name)
                pdf_plot.export(
                    fname="{}/pdf_{}_1D.png".format(self.paths.pdf_path, param_name)
                )

        plt.close()

        plot_pdf_triangle = conf.instance.visualize_plots.get("pdf", "triangle", bool)

        if plot_pdf_triangle:

            try:
                pdf_plot.triangle_plot(roots=self.pdf)
                pdf_plot.export(fname="{}/pdf_triangle.png".format(self.paths.pdf_path))
            except Exception as e:
                print(type(e))
                print(
                    "The PDF triangle of this non-linear search could not be plotted. This is most likely due to a "
                    "lack of smoothness in the sampling of parameter space. Sampler further by decreasing the "
                    "parameter evidence_tolerance."
                )

        plt.close()


class MultiNestSamples(NestedSamplerSamples):
    @property
    def pdf(self):
        """An interface to *GetDist* which can be used for analysing and visualizing the non-linear search chains.

        *GetDist* can only be used when chains are converged enough to provide a smooth PDF and this convergence is
        checked using the *pdf_converged* bool before *GetDist* is called.

        https://github.com/cmbant/getdist
        https://getdist.readthedocs.io/en/latest/

        For *MultiNest*, chains are passed to *GetDist* via the multinest.txt file, which contains the physical model
        parameters of every accepted sample and its sampling probabilitiy which is used as the weight.
        """

        import getdist

        try:
            return getdist.mcsamples.loadMCSamples(
                self.paths.backup_path + "/multinest"
            )
        except IOError or OSError or ValueError or IndexError:
            raise Exception

    @property
    def pdf_converged(self) -> bool:
        """ To analyse and visualize chains using *GetDist*, the analysis must be sufficiently converged to produce
        smooth enough PDF for analysis. This property checks whether the non-linear search's chains are sufficiently
        converged for *GetDist* use.

        For *MultiNest*, during initial sampling one accepted live point typically has > 99% of the probabilty as its
        log_likelihood is significantly higher than all other points. Convergence is only achieved late in sampling when
        all live points have similar log_likelihood and sampling probabilities."""
        try:
            densities_1d = list(
                map(lambda p: self.pdf.get1DDensity(p), self.pdf.getParamNames().names)
            )

            if len(densities_1d) == 0:
                return False

            return True
        except Exception:
            return False

    @property
    def number_live_points(self) -> int:
        """The number of live points used by the nested sampler."""
        return len(self.phys_live_points)

    @property
    def total_samples(self) -> int:
        """The total number of samples performed by the non-linear search.

        For MulitNest, this includes all accepted and rejected samples, and is loaded from the "multinestresume" file.
        """
        resume = open(self.paths.file_resume)

        resume.seek(1)
        resume.read(19)
        return int(resume.read(8))

    @property
    def total_accepted_samples(self) -> int:
        """The total number of accepted samples performed by the non-linear search.

        For MulitNest, this is loaded from the "multinestresume" file.
        """
        resume = open(self.paths.file_resume)

        resume.seek(1)
        resume.read(8)
        return int(resume.read(10))

    @property
    def acceptance_ratio(self) -> float:
        """The ratio of accepted samples to total samples."""
        return self.total_accepted_samples / self.total_samples

    @property
    def max_log_posterior(self) -> float:
        """The maximum log likelihood value of the non-linear search, corresponding to the best-fit model.

        For MultiNest, this is read from the "multinestsummary.txt" file. """
        try:
            return self.read_list_of_results_from_summary_file(
                number_entries=2, offset=112
            )[1]
        except FileNotFoundError:
            return max([point[-1] for point in self.phys_live_points])

    @property
    def log_evidence(self) -> float:
        """The Bayesian log evidence estimated by the nested sampling algorithm.

        For MultiNest, this is read from the "multinestsummary.txt" file."""
        try:
            return self.read_list_of_results_from_summary_file(
                number_entries=2, offset=112
            )[0]
        except FileNotFoundError:
            return None

    @property
    def max_log_likelihood_index(self) -> int:
        """The index of the accepted sample with the highest log likelihood, e.g. that of best-fit / most_likely model."""
        return int(np.argmax([point[-1] for point in self.phys_live_points]))

    @property
    def max_log_likelihood_vector(self) -> [float]:
        """ The best-fit model sampled by the non-linear search (corresponding to the maximum log likelihood), returned
        as a list of values.

        The vector is read from the MulitNest file "multinestsummary.txt, which stores the parameters of the most
        likely model in the second half of entries."""
        try:
            return self.read_list_of_results_from_summary_file(
                number_entries=self.model.prior_count, offset=56
            )
        except FileNotFoundError:
            return self.phys_live_points[self.max_log_likelihood_index][0:-1]

    @property
    def most_probable_vector(self) -> [float]:
        """ The median of the probability density function (PDF) of every parameter marginalized in 1D, returned
        as a list of values.

        The vector is read from the MulitNest file "multinestsummary.txt, which stores the parameters of the most
        probable model in the first half of entries.
        """
        try:
            return self.read_list_of_results_from_summary_file(
                number_entries=self.model.prior_count, offset=0
            )
        except FileNotFoundError:
            return self.max_log_likelihood_vector

    def vector_at_sigma(self, sigma) -> [float]:
        """ The value of every parameter marginalized in 1D at an input sigma value of its probability density function
        (PDF), returned as two lists of values corresponding to the lower and upper values parameter values.

        For example, if sigma is 1.0, the marginalized values of every parameter at 31.7% and 68.2% percentiles of each
        PDF is returned.

        This does not account for covariance between parameters. For example, if two parameters (x, y) are degenerate
        whereby x decreases as y gets larger to give the same PDF, this function will still return both at their
        upper values. Thus, caution is advised when using the function to reperform a model-fits.

        For *MultiNest*, this is estimated using *GetDist* if the chains have converged, by sampling the density
        function at an input PDF %. If not converged, a crude estimate using the range of values of the current
        physical live points is used.

        Parameters
        ----------
        sigma : float
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF)."""
        limit = math.erf(0.5 * sigma * math.sqrt(2))

        if self.pdf_converged:
            densities_1d = list(
                map(lambda p: self.pdf.get1DDensity(p), self.pdf.getParamNames().names)
            )

            return list(map(lambda p: p.getLimits(limit), densities_1d))
        else:

            parameters_min = [
                min(self.phys_live_points_of_param(param_index=param_index))
                for param_index in range(self.model.prior_count)
            ]
            parameters_max = [
                max(self.phys_live_points_of_param(param_index=param_index))
                for param_index in range(self.model.prior_count)
            ]

            return [
                (parameters_min[index], parameters_max[index])
                for index in range(len(parameters_min))
            ]

    def vector_from_sample_index(self, sample_index) -> [float]:
        """The model parameters of an individual sample of the non-linear search.

        Parameters
        ----------
        sample_index : int
            The index of the sample in the non-linear search, e.g. 0 gives the first sample.
        """
        return list(self.pdf.samples[sample_index])

    def weight_from_sample_index(self, sample_index) -> float:
        """The weight of an individual sample of the non-linear search.

        Parameters
        ----------
        sample_index : int
            The index of the sample in the non-linear search, e.g. 0 gives the first sample.
        """
        return self.pdf.weights[sample_index]

    def log_likelihood_from_sample_index(self, sample_index) -> float:
        """The log likelihood of an individual sample of the non-linear search.

        NOTE: GetDist reads the log likelihood from the weighted_sample.txt file (column 2), which are defined as \
        -2.0*log_likelihood. This routine converts these back to log_likelihood.

        Parameters
        ----------
        sample_index : int
            The index of the sample in the non-linear search, e.g. 0 gives the first sample.
        """
        return -0.5 * self.pdf.loglikes[sample_index]

    @property
    def phys_live_points(self) -> [[float]]:
        """Open the MultiNest "multinestphys_live_points" file, read it and return all physical live point models as
        a list of 1D lists."""
        phys_live = open(self.paths.file_phys_live)

        live_points = 0
        for line in phys_live:
            live_points += 1

        phys_live.seek(0)

        phys_live_points = []

        for line in range(live_points):
            vector = []
            for param in range(self.model.prior_count + 1):
                vector.append(float(phys_live.read(28)))
            phys_live.readline()
            phys_live_points.append(vector)

        phys_live.close()

        return phys_live_points

    def phys_live_points_of_param(self, param_index) -> [float]:
        """Return all parameter values of a given parameter for all of the current physical live points.

        These are read from the "multinestphys_live_points" file."""
        return [point[param_index] for point in self.phys_live_points]

    def read_list_of_results_from_summary_file(self, number_entries, offset) -> [float]:
        """Read a list of results from the "multinestsummary.txt" file, which stores information on the MulitNest run
        incuding the most likely and most probable models.

        This file stores the results as a text file where different columns correspnd to different models."""

        summary = open(self.paths.file_summary)
        summary.read(2 + offset * self.model.prior_count)
        vector = []
        for param in range(number_entries):
            vector.append(float(summary.read(28)))

        summary.close()

        return vector


class DynestySamples(NestedSamplerSamples):
    @property
    def sampler(self):
        """The pickled instance of the *Dynesty* sampler, which provides access to all accept sample likelihoods,
        weights, etc. of the non-linear search.

        The sampler is described in the "Results" section at https://dynesty.readthedocs.io/en/latest/quickstart.html"""
        with open("{}/{}.pickle".format(self.paths.chains_path, "dynesty"), "rb") as f:
            return pickle.load(f)

    @property
    def results(self):
        """Convenience method to the pickled sample's results."""
        with open("{}/{}.pickle".format(self.paths.chains_path, "results"), "rb") as f:
            return pickle.load(f)

    @property
    def pdf(self):
        """An interface to *GetDist* which can be used for analysing and visualizing the non-linear search chains.

        *GetDist* can only be used when chains are converged enough to provide a smooth PDF and this convergence is
        checked using the *pdf_converged* bool before *GetDist* is called.

        https://github.com/cmbant/getdist
        https://getdist.readthedocs.io/en/latest/

        For *Dynesty*, chains are passed to *GetDist* using the pickled sapler instance, which contains the physical
        model parameters of every accepted sample, their likelihoods and weights.
        """
        import getdist

        return getdist.mcsamples.MCSamples(
            samples=self.results.samples,
            weights=self.results.logwt,
            loglikes=self.results.logl,
        )

    @property
    def pdf_converged(self) -> bool:
        """ To analyse and visualize chains using *GetDist*, the analysis must be sufficiently converged to produce
        smooth enough PDF for analysis. This property checks whether the non-linear search's chains are sufficiently
        converged for *GetDist* use.

        For *Dynesty*, during initial sampling one accepted live point typically has > 99% of the probabilty as its
        log_likelihood is significantly higher than all other points. Convergence is only achieved late in sampling when
        all live points have similar log_likelihood and sampling probabilities."""
        try:
            densities_1d = list(
                map(lambda p: self.pdf.get1DDensity(p), self.pdf.getParamNames().names)
            )

            if densities_1d == []:
                return False

            return True
        except Exception:
            return False

    @property
    def number_live_points(self) -> int:
        """The number of live points used by the nested sampler."""
        return int(np.sum(self.results.nlive))

    @property
    def total_samples(self) -> int:
        """The total number of samples performed by the non-linear search.

        For Dynesty, this includes all accepted and rejected samples, and is loaded from the sampler pickle.
        """
        return int(np.sum(self.results.ncall))

    @property
    def total_accepted_samples(self) -> int:
        """The total number of accepted samples performed by the non-linear search.

        For Dynesty, this is loaded from the pickled sampler.
        """
        return self.results.niter

    @property
    def acceptance_ratio(self) -> float:
        """The ratio of accepted samples to total samples."""
        return self.total_accepted_samples / self.total_samples

    @property
    def max_log_likelihood(self) -> float:
        """The maximum log likelihood value of the non-linear search, corresponding to the best-fit model.

        For Dynesty, this is computed from the pickled sampler's list of all log likelihood values."""
        return np.max(self.results.logl)

    @property
    def max_log_likelihood_vector(self) -> [float]:
        """ The best-fit model sampled by the non-linear search (corresponding to the maximum log likelihood), returned
        as a list of values.

        The vector is read from the pickled sampler instance, by first locating the index corresponding to the highest
        log_likelihood accepted sample."""
        return self.results.samples[self.max_log_likelihood_index]

    @property
    def log_evidence(self) -> float:
        """The Bayesian log evidence estimated by the nested sampling algorithm.

        For Dynesty, this is computed from the pickled sample's list of all log evidence estimates."""
        return np.max(self.results.logz)

    @property
    def most_probable_vector(self) -> [float]:
        """ The median of the probability density function (PDF) of every parameter marginalized in 1D, returned
        as a list of values.

        If the chains are sufficiently converged this is estimated by passing the accepted samples to *GetDist*, else
        a crude estimate using the mean value of all accepted samples is used."""
        if self.pdf_converged:
            return self.pdf.getMeans()
        else:
            return list(np.mean(self.results.samples, axis=0))

    def vector_from_sample_index(self, sample_index) -> [float]:
        """The model parameters of an individual sample of the non-linear search.

        Parameters
        ----------
        sample_index : int
            The index of the sample in the non-linear search, e.g. 0 gives the first sample.
        """
        return self.results.samples[sample_index]

    def weight_from_sample_index(self, sample_index) -> float:
        """The weight of an individual sample of the non-linear search.

        Parameters
        ----------
        sample_index : int
            The index of the sample in the non-linear search, e.g. 0 gives the first sample.
        """
        return self.results.logwt[sample_index]

    def log_likelihood_from_sample_index(self, sample_index) -> float:
        """The log likelihood of an individual sample of the non-linear search.

        Parameters
        ----------
        sample_index : int
            The index of the sample in the non-linear search, e.g. 0 gives the first sample.
        """
        return self.results.logl[sample_index]
