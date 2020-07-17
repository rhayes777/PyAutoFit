import csv
import math
from typing import List

import corner
import emcee
import numpy as np

from autofit.mapper.model import ModelInstance
from autofit.mapper.model_mapper import ModelMapper
from autofit.text import model_text


class OptimizerSamples:
    def __init__(
            self,
            model: ModelMapper,
            parameters: List[List[float]],
            log_likelihoods: List[float],
            log_priors: List[float],
            weights: List[float],
            time: float = None
    ):
        """The *Samples* of a non-linear search, specifically the samples of an search which only provides
        information on the global maximum likelihood solutions, but does not map-out the posterior and thus does
        not provide information on parameter errors.

        Parameters
        ----------
        model : af.ModelMapper
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        """
        self.model = model
        self.total_samples = len(log_likelihoods)
        self.parameters = parameters
        self.log_likelihoods = log_likelihoods
        self.log_priors = log_priors
        self.log_posteriors = [
            lh + prior for lh, prior in zip(log_likelihoods, log_priors)
        ]
        self.weights = weights
        self.time = time

    @property
    def parameter_names(self):
        return self.model.parameter_names

    @property
    def parameters_extract(self):
        return [[params[i] for params in self.parameters] for i in range(self.model.prior_count)]

    @property
    def _headers(self) -> List[str]:
        """
        Headers for the samples table
        """
        return self.parameter_names + ["log_likelihood", "log_prior", "log_posterior", "weights"]

    @property
    def _rows(self) -> List[List[float]]:
        """
        Rows in the samples table
        """
        for index, row in enumerate(self.parameters):
            yield row + [
                self.log_likelihoods[index],
                self.log_priors[index],
                self.log_posteriors[index],
                self.weights[index]
            ]

    def write_table(self, filename: str):
        """
        Write a table of parameters, posteriors, priors and likelihoods

        Parameters
        ----------
        filename
            Where the table is to be written
        """
        with open(filename, "w+") as f:
            writer = csv.writer(f)
            writer.writerow(self._headers)
            for row in self._rows:
                writer.writerow(row)

    @property
    def parameter_labels(self):
        return model_text.parameter_labels_from_model(model=self.model)

    @property
    def max_log_likelihood_index(self) -> int:
        """The index of the sample with the highest log likelihood."""
        return int(np.argmax(self.log_likelihoods))

    @property
    def max_log_likelihood_vector(self) -> [float]:
        """ The parameters of the maximum log likelihood sample of the non-linear search returned as a list of values."""
        return self.parameters[self.max_log_likelihood_index]

    @property
    def max_log_likelihood_instance(self) -> ModelInstance:
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
    def max_log_posterior_instance(self) -> ModelInstance:
        """  The parameters of the maximum log posterior sample of the non-linear search returned as a model instance."""
        return self.model.instance_from_vector(vector=self.max_log_posterior_vector)

    def gaussian_priors_at_sigma(self, sigma) -> [list]:
        """*GaussianPrior*s of every parameter used to link its inferred values and errors to priors used to sample the
        same (or similar) parameters in a subsequent phase, where:

         - The mean is given by maximum log likelihood model values.
         - Their errors are omitted, as this information is not available from an search. When these priors are
           used to link to another phase, it will thus automatically use the prior config values.

        Parameters
        -----------
        sigma : float
            The sigma limit within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the \
            PDF).
        """
        return list(map(lambda vector: (vector, 0.0), self.max_log_likelihood_vector))

    def instance_from_sample_index(self, sample_index) -> ModelInstance:
        """The parameters of an individual saple of the non-linear search, returned as a model instance.

        Parameters
        -----------
        sample_index : int
            The sample index of the weighted sample to return.
        """
        return self.model.instance_from_vector(vector=self.parameters[sample_index])


class PDFSamples(OptimizerSamples):
    def __init__(
            self,
            model: ModelMapper,
            parameters: List[List[float]],
            log_likelihoods: List[float],
            log_priors: List[float],
            weights: List[float],
            unconverged_sample_size: int = 100,
            time: float = None,
    ):
        """The *Samples* of a non-linear search, specifically the samples of a non-linear search which maps out the
        posterior of parameter space and thus does provide information on parameter errors.

        Parameters
        ----------
        model : af.ModelMapper
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        """

        super().__init__(
            model=model,
            parameters=parameters,
            log_likelihoods=log_likelihoods,
            log_priors=log_priors,
            weights=weights,
            time=time
        )

        self._unconverged_sample_size = unconverged_sample_size

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
    def pdf_converged(self) -> bool:
        """ To analyse and visualize samples the analysis must be sufficiently converged to produce smooth enough
        PDF for error estimate and PDF generation.

        This property checks whether the non-linear search's samples are sufficiently converged for this, by checking
        if one sample's weight contains > 99% of the weight. If this is the case, it implies the convergence necessary
        for error estimate and visualization has not been met.

        This does not necessarily imply the non-linear search has converged overall, only that errors and visualization
        can be performed numerically.."""
        if np.max(self.weights) > 0.99:
            return False
        return True

    @property
    def median_pdf_vector(self) -> [float]:
        """ The median of the probability density function (PDF) of every parameter marginalized in 1D, returned
        as a list of values."""
        if self.pdf_converged:
            return [corner.quantile(x=params, q=0.5, weights=self.weights)[0] for params in self.parameters_extract]
        return self.max_log_likelihood_vector

    @property
    def median_pdf_instance(self) -> ModelInstance:
        """ The median of the probability density function (PDF) of every parameter marginalized in 1D, returned
        as a model instance."""
        return self.model.instance_from_vector(vector=self.median_pdf_vector)

    def vector_at_sigma(self, sigma) -> [float]:
        """ The value of every parameter marginalized in 1D at an input sigma value of its probability density function
        (PDF), returned as two lists of values corresponding to the lower and upper values parameter values.

        For example, if sigma is 1.0, the marginalized values of every parameter at 31.7% and 68.2% percentiles of each
        PDF is returned.

        This does not account for covariance between parameters. For example, if two parameters (x, y) are degenerate
        whereby x decreases as y gets larger to give the same PDF, this function will still return both at their
        upper values. Thus, caution is advised when using the function to reperform a model-fits.

        For *Dynesty*, this is estimated using *GetDist* if the samples have converged, by sampling the density
        function at an input PDF %. If not converged, a crude estimate using the range of values of the current
        physical live points is used.

        Parameters
        ----------
        sigma : float
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF)."""

        if self.pdf_converged:
            limit = math.erf(0.5 * sigma * math.sqrt(2))

            lower_errors = [corner.quantile(x=params, q=1.0 - limit, weights=self.weights)[0] for params in
                            self.parameters_extract]

            upper_errors = [corner.quantile(x=params, q=limit, weights=self.weights)[0] for params in
                            self.parameters_extract]

            return [(lower, upper) for lower, upper in zip(lower_errors, upper_errors)]

        parameters_min = list(
            np.min(self.parameters[-self.unconverged_sample_size:], axis=0)
        )
        parameters_max = list(
            np.max(self.parameters[-self.unconverged_sample_size:], axis=0)
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

    def instance_at_sigma(self, sigma) -> ModelInstance:
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

    def instance_at_upper_sigma(self, sigma) -> ModelInstance:
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

    def instance_at_lower_sigma(self, sigma) -> ModelInstance:
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
                lambda upper, median_pdf: upper - median_pdf,
                uppers,
                self.median_pdf_vector,
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
                lambda lower, median_pdf: median_pdf - lower,
                lowers,
                self.median_pdf_vector,
            )
        )

    def error_instance_at_sigma(self, sigma) -> ModelInstance:
        """ The error of every parameter marginalized in 1D at an input sigma value of its probability density function
        (PDF), returned as a list of model instances corresponding to the lower and upper errors.

        See *vector_at_sigma* for a full description of how the parameters at sigma are computed.

        Parameters
        ----------
        sigma : float
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF)."""
        return self.model.instance_from_vector(
            vector=self.error_vector_at_sigma(sigma=sigma), assert_priors_in_limits=False
        )

    def error_instance_at_upper_sigma(self, sigma) -> ModelInstance:
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
            vector=self.error_vector_at_upper_sigma(sigma=sigma), assert_priors_in_limits=False
        )

    def error_instance_at_lower_sigma(self, sigma) -> ModelInstance:
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
            vector=self.error_vector_at_lower_sigma(sigma=sigma), assert_priors_in_limits=False
        )

    def gaussian_priors_at_sigma(self, sigma) -> [list]:
        """*GaussianPrior*s of every parameter used to link its inferred values and errors to priors used to sample the
        same (or similar) parameters in a subsequent phase, where:

         - The mean is given by their most-probable values (using *median_pdf_vector*).
         - Their errors are computed at an input sigma value (using *errors_at_sigma*).

        Parameters
        -----------
        sigma : float
            The sigma limit within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the \
            PDF).
        """

        means = self.median_pdf_vector
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

    def offset_vector_from_input_vector(self, input_vector) -> [float]:
        """ The values of an input_vector offset by the *median_pdf_vector* (the PDF medians).

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
                lambda input, median_pdf: median_pdf - input,
                input_vector,
                self.median_pdf_vector,
            )
        )

    def output_pdf_plots(self):
        """Output plots of the probability density functions of the non-linear seach.

        This uses *GetDist* to plot:

         - The marginalize 1D PDF of every parameter.
         - The marginalized 2D PDF of every parameter pair.
         - A Triangle plot of the 2D and 1D PDF's.
         """

        pass

        # import getdist.plots
        # import matplotlib
        #
        # backend = conf.instance.visualize_general.get("general", "backend", str)
        # if not backend in "default":
        #     matplotlib.use(backend)
        # if conf.instance.general.get("hpc", "hpc_mode", bool):
        #     matplotlib.use("Agg")
        # import matplotlib.pyplot as plt
        #
        # pdf_plot = getdist.plots.GetDistPlotter()
        #
        # plot_pdf_1d_params = conf.instance.visualize_plots.get("pdf", "1d_params", bool)
        #
        # if plot_pdf_1d_params:
        #
        #     for param_name in self.model.parameter_names:
        #         pdf_plot.plot_1d(roots=self.getdist_samples, param=param_name)
        #         pdf_plot.export(
        #             fname="{}/pdf_{}_1D.png".format(self.paths.pdf_path, param_name)
        #         )
        #
        # plt.close()
        #
        # plot_pdf_triangle = conf.instance.visualize_plots.get("pdf", "triangle", bool)
        #
        # if plot_pdf_triangle:
        #
        #     try:
        #         pdf_plot.triangle_plot(roots=self.getdist_samples)
        #         pdf_plot.export(fname="{}/pdf_triangle.png".format(self.paths.pdf_path))
        #     except Exception as e:
        #         logger.exception(e)
        #         print(
        #             "The PDF triangle of this non-linear search could not be plotted. This is most likely due to a "
        #             "lack of smoothness in the sampling of parameter space. Sampler further by decreasing the "
        #             "parameter evidence_tolerance."
        #         )
        #
        # plt.close()


class MCMCSamples(PDFSamples):
    def __init__(
            self,
            model: ModelMapper,
            parameters: List[List[float]],
            log_likelihoods: List[float],
            log_priors: List[float],
            weights: List[float],
            auto_correlation_times: np.ndarray,
            auto_correlation_check_size: int,
            auto_correlation_required_length: int,
            auto_correlation_change_threshold: float,
            total_walkers: int,
            total_steps: int,
            backend: emcee.backends.HDFBackend,
            unconverged_sample_size: int = 100,
            time: float = None
    ):
        """
        Attributes
        ----------
        total_walkers : int
            The total number of walkers used by this MCMC non-linear search.
        total_steps : int
            The total number of steps taken by each walker of this MCMC non-linear search (the total samples is equal
            to the total steps * total walkers).
        """

        super().__init__(
            model=model,
            parameters=parameters,
            log_likelihoods=log_likelihoods,
            log_priors=log_priors,
            weights=weights,
            unconverged_sample_size=unconverged_sample_size,
            time=time
        )

        self.total_walkers = total_walkers
        self.total_steps = total_steps
        self.auto_correlation_times = auto_correlation_times
        self.auto_correlation_check_size = auto_correlation_check_size
        self.auto_correlation_required_length = auto_correlation_required_length
        self.auto_correlation_change_threshold = auto_correlation_change_threshold
        self.log_evidence = None
        self.backend = backend

    @property
    def pdf_converged(self):
        """ To analyse and visualize samples using *GetDist*, the analysis must be sufficiently converged to produce
        smooth enough PDF for analysis. This property checks whether the non-linear search's samples are sufficiently
        converged for *GetDist* use.

        Emcee samples can be analysed by GetDist irrespective of how long the sampler has run, albeit low run times
        will likely produce inaccurate results."""
        try:
            self.samples_after_burn_in
            return True
        except ValueError:
            return False

    @property
    def samples_after_burn_in(self) -> [list]:
        """The emcee samples with the initial burn-in samples removed.

        The burn-in period is estimated using the auto-correlation times of the parameters."""
        discard = int(3.0 * np.max(self.auto_correlation_times))
        thin = int(np.max(self.auto_correlation_times) / 2.0)
        return self.backend.get_chain(discard=discard, thin=thin, flat=True)

    @property
    def previous_auto_correlation_times(self) -> [float]:
        return emcee.autocorr.integrated_time(
            x=self.backend.get_chain()[: -self.auto_correlation_check_size, :, :], tol=0
        )

    @property
    def relative_auto_correlation_times(self) -> [float]:
        return (
                np.abs(self.previous_auto_correlation_times - self.auto_correlation_times)
                / self.auto_correlation_times
        )

    @property
    def converged(self) -> bool:
        """Whether the emcee samples have converged on a solution or if they are still in a burn-in period, based on the
        auto correlation times of parameters."""
        converged = np.all(
            self.auto_correlation_times * self.auto_correlation_required_length
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
    def median_pdf_vector(self) -> [float]:
        """ The median of the probability density function (PDF) of every parameter marginalized in 1D, returned
        as a list of values.

        This is computed by binning all sampls after burn-in into a histogram and take its median (e.g. 50%) value. """
        if self.pdf_converged:
            return [
                float(np.percentile(self.samples_after_burn_in[:, i], [50]))
                for i in range(self.model.prior_count)
            ]

        return self.max_log_likelihood_vector

    def vector_at_sigma(self, sigma) -> [float]:
        """ The value of every parameter marginalized in 1D at an input sigma value of its probability density function
        (PDF), returned as two lists of values corresponding to the lower and upper values parameter values.

        For example, if sigma is 1.0, the marginalized values of every parameter at 31.7% and 68.2% percentiles of each
        PDF is returned.

        This does not account for covariance between parameters. For example, if two parameters (x, y) are degenerate
        whereby x decreases as y gets larger to give the same PDF, this function will still return both at their
        upper values. Thus, caution is advised when using the function to reperform a model-fits.

        For *Emcee*, if the samples have converged this is estimated by binning the samples after burn-in into a
        histogram and taking the parameter values at the input PDF %.

        Parameters
        ----------
        sigma : float
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF)."""
        limit = math.erf(0.5 * sigma * math.sqrt(2))

        if self.pdf_converged:
            samples = self.samples_after_burn_in

            return [
                tuple(
                    np.percentile(samples[:, i], [100.0 * (1.0 - limit), 100.0 * limit])
                )
                for i in range(self.model.prior_count)
            ]

        parameters_min = list(
            np.min(self.parameters[-self.unconverged_sample_size:], axis=0)
        )
        parameters_max = list(
            np.max(self.parameters[-self.unconverged_sample_size:], axis=0)
        )

        return [
            (parameters_min[index], parameters_max[index])
            for index in range(len(parameters_min))
        ]


class NestSamples(PDFSamples):
    def __init__(
            self,
            model: ModelMapper,
            parameters: List[List[float]],
            log_likelihoods: List[float],
            log_priors: List[float],
            weights: List[float],
            number_live_points: int,
            log_evidence: float,
            total_samples: int,
            unconverged_sample_size: int = 100,
            time: float = None
    ):
        """The *Output* classes in **PyAutoFit** provide an interface between the results of a non-linear search (e.g.
        as files on your hard-disk) and Python.

        For example, the output class can be used to load an instance of the best-fit model, get an instance of any
        individual sample by the non-linear search and return information on the likelihoods, errors, etc.

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
            parameters=parameters,
            log_likelihoods=log_likelihoods,
            log_priors=log_priors,
            weights=weights,
            unconverged_sample_size=unconverged_sample_size,
            time=time
        )

        self.number_live_points = number_live_points
        self.total_samples = total_samples
        self.log_evidence = log_evidence

    @property
    def total_accepted_samples(self) -> int:
        """The total number of accepted samples performed by the nested sampler.
        """
        return len(self.log_likelihoods)

    @property
    def acceptance_ratio(self) -> float:
        """The ratio of accepted samples to total samples."""
        return self.total_accepted_samples / self.total_samples
