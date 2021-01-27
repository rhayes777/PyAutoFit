import csv
import json
import math
from typing import List

import numpy as np

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.mapper.model import ModelInstance
from autofit.mapper.model_mapper import ModelMapper
from autofit.tools import util

class Sample:
    def __init__(
            self,
            log_likelihood: float,
            log_prior: float,
            weights: float,
            **kwargs
    ):
        """
        One sample taken during a search

        Parameters
        ----------
        log_likelihood
            The likelihood associated with this instance
        log_prior
            A logarithmic prior of the instance
        weights
        kwargs
            Dictionary mapping model paths to values for the sample
        """
        self.log_likelihood = log_likelihood
        self.log_prior = log_prior
        self.weights = weights
        self.kwargs = kwargs

    @property
    def log_posterior(self) -> float:
        """
        Compute the posterior
        """
        return self.log_likelihood + self.log_prior

    def parameters_for_model(
            self,
            model: AbstractPriorModel,
            paths=None,
    ) -> List[float]:
        """
        Values for instantiating a model, in the same order as priors
        from the model.

        Parameters
        ----------
        model
            The model from which this was a sample

        Returns
        -------
        A list of physical values
        """

        if paths is None:
            paths = model.model_component_and_parameter_names

        return [
            self.kwargs[path]
            for path
            in paths
        ]

        #
        #
        # # TODO : The commented out code and try / except are equivalent, except the latter adds a _value string to a
        # # TODO : path if the first is not found.
        # # TODO: This is due to horrific backwards compaitbility issues which we currently cannot remove,
        # # TODO: so it'll have to do for now. One day we'll delete this.
        #
        # # return [
        # #     self.kwargs[path]
        # #     for path
        # #     in paths
        # # ]
        #
        # parameters_bc = []
        #
        # print(self.kwargs)
        #
        # for path in paths:
        #     try:
        #         parameters_bc.append(self.kwargs[path])
        #     except KeyError:
        #         parameters_bc.append(self.kwargs[f"{path}_value"])
        #
        # return parameters_bc

    @classmethod
    def from_lists(
            cls,
            model: AbstractPriorModel,
            parameters: List[List[float]],
            log_likelihoods: List[float],
            log_priors: List[float],
            weights: List[float]
    ) -> List["Sample"]:
        """
        Convenience method to create a list of samples
        from lists of contained values

        Parameters
        ----------
        model
        parameters
        log_likelihoods
        log_priors
        weights

        Returns
        -------
        A list of samples
        """
        samples = list()

        # Another speed up.
        model_component_and_parameter_names = model.model_component_and_parameter_names

        for params, log_likelihood, log_prior, weight in zip(
                parameters,
                log_likelihoods,
                log_priors,
                weights
        ):
            arg_dict = {
                t: param
                for t, param
                in zip(
                    model_component_and_parameter_names,
                    params
                )
            }

            samples.append(
                cls(
                    log_likelihood=log_likelihood,
                    log_prior=log_prior,
                    weights=weight,
                    **arg_dict
                )
            )
        return samples

    def instance_for_model(self, model: AbstractPriorModel):
        """
        Create an instance from this sample for a model

        Parameters
        ----------
        model
            The model the this sample was taken from

        Returns
        -------
        The instance corresponding to this sample
        """
        try:
            return model.instance_from_vector(
                self.parameters_for_model(model)
            )
        except KeyError:
            paths = model.model_component_and_parameter_names
            paths = util.convert_paths_for_backwards_compatibility(paths=paths, kwargs=self.kwargs)
            return model.instance_from_vector(
                self.parameters_for_model(model, paths)
            )


def load_from_table(filename: str) -> List[Sample]:
    """
    Load samples from a table

    Parameters
    ----------
    filename
        The path to a CSV file

    Returns
    -------
    A list of samples, one for each row in the CSV
    """
    samples = list()

    with open(filename, "r+", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            samples.append(
                Sample(
                    **{
                        header: float(value)
                        for header, value
                        in zip(
                            headers,
                            row
                        )
                    }
                )
            )

    return samples


class OptimizerSamples:
    def __init__(
            self,
            model: ModelMapper,
            samples: List[Sample],
            time: float = None,
    ):
        """The `Samples` of a non-linear search, specifically the samples of an search which only provides
        information on the global maximum likelihood solutions, but does not map-out the posterior and thus does
        not provide information on parameter errors.

        Parameters
        ----------
        model : af.ModelMapper
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        """
        self.model = model
        self.samples = samples
        self.time = time

    @property
    def parameters(self):

        paths = self.model.model_component_and_parameter_names

        try:
            return [
                sample.parameters_for_model(
                    self.model, paths
                )
                for sample in self.samples
            ]
        except KeyError:
            paths = util.convert_paths_for_backwards_compatibility(paths=paths, kwargs=self.samples[0].kwargs)
            return [
                sample.parameters_for_model(
                    self.model, paths
                )
                for sample in self.samples
            ]

    @property
    def total_samples(self):
        return len(self.samples)

    @property
    def weights(self):
        return [
            sample.weights
            for sample
            in self.samples
        ]

    @property
    def log_likelihoods(self):
        return [
            sample.log_likelihood
            for sample
            in self.samples
        ]

    @property
    def log_posteriors(self):
        return [
            sample.log_posterior
            for sample
            in self.samples
        ]

    @property
    def log_priors(self):
        return [
            sample.log_prior
            for sample
            in self.samples
        ]

    @property
    def parameters_extract(self):
        return [
            [params[i] for params in self.parameters]
            for i in range(self.model.prior_count)
        ]

    @property
    def _headers(self) -> List[str]:
        """
        Headers for the samples table
        """

        return self.model.model_component_and_parameter_names + [
            "log_likelihood",
            "log_prior",
            "log_posterior",
            "weights",
        ]

    @property
    def _rows(self) -> List[List[float]]:
        """
        Rows in the samples table
        """

        log_likelihoods = self.log_likelihoods
        log_priors = self.log_priors
        log_posteriors = self.log_posteriors
        weights = self.weights

        for index, row in enumerate(self.parameters):
            yield row + [
                log_likelihoods[index],
                log_priors[index],
                log_posteriors[index],
                weights[index],
            ]

    def write_table(self, filename: str):
        """
        Write a table of parameters, posteriors, priors and likelihoods

        Parameters
        ----------
        filename
            Where the table is to be written
        """

        with open(filename, "w+", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self._headers)
            for row in self._rows:
                writer.writerow(row)

    def info_to_json(self, filename):

        info = {}

        with open(filename, 'w') as outfile:
            json.dump(info, outfile)

    @property
    def max_log_likelihood_sample(self) -> Sample:
        """The index of the sample with the highest log likelihood."""
        most_likely_sample = None
        for sample in self.samples:
            if most_likely_sample is None or sample.log_likelihood > most_likely_sample.log_likelihood:
                most_likely_sample = sample
        return most_likely_sample

    @property
    def max_log_likelihood_vector(self) -> [float]:
        """ The parameters of the maximum log likelihood sample of the `NonLinearSearch` returned as a list of values."""
        return self.max_log_likelihood_sample.parameters_for_model(
            self.model
        )

    @property
    def max_log_likelihood_instance(self) -> ModelInstance:
        """  The parameters of the maximum log likelihood sample of the `NonLinearSearch` returned as a model instance."""
        return self.max_log_likelihood_sample.instance_for_model(
            self.model
        )

    @property
    def max_log_posterior_index(self) -> int:
        """The index of the sample with the highest log posterior."""
        return int(np.argmax(self.log_posteriors))

    @property
    def max_log_posterior_vector(self) -> [float]:
        """ The parameters of the maximum log posterior sample of the `NonLinearSearch` returned as a list of values."""
        return self.parameters[self.max_log_posterior_index]

    @property
    def max_log_posterior_instance(self) -> ModelInstance:
        """  The parameters of the maximum log posterior sample of the `NonLinearSearch` returned as a model instance."""
        return self.model.instance_from_vector(vector=self.max_log_posterior_vector)

    def gaussian_priors_at_sigma(self, sigma) -> [list]:
        """`GaussianPrior`s of every parameter used to link its inferred values and errors to priors used to sample the
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
            samples: List[Sample],
            unconverged_sample_size: int = 100,
            time: float = None,
    ):
        """The `Samples` of a non-linear search, specifically the samples of a `NonLinearSearch` which maps out the
        posterior of parameter space and thus does provide information on parameter errors.

        Parameters
        ----------
        model : af.ModelMapper
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        """

        super().__init__(
            model=model,
            samples=samples,
            time=time,
        )

        self._unconverged_sample_size = unconverged_sample_size

    @classmethod
    def from_table(self, filename: str, model, number_live_points=None):
        """
        Write a table of parameters, posteriors, priors and likelihoods

        Parameters
        ----------
        filename
            Where the table is to be written
        """

        samples = load_from_table(
            filename=filename
        )

        return OptimizerSamples(
            model=model,
            samples=samples
        )

    @property
    def unconverged_sample_size(self):
        """If a set of samples are unconverged, alternative methods to compute their means, errors, etc are used as
        an alternative to corner.py.

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

        This does not necessarily imply the `NonLinearSearch` has converged overall, only that errors and visualization
        can be performed numerically.."""
        if np.max(self.weights) > 0.99:
            return False
        return True

    @property
    def median_pdf_vector(self) -> [float]:
        """ The median of the probability density function (PDF) of every parameter marginalized in 1D, returned
        as a list of values."""
        if self.pdf_converged:
            return [
                quantile(x=params, q=0.5, weights=self.weights)[0]
                for params in self.parameters_extract
            ]
        return self.max_log_likelihood_vector

    @property
    def median_pdf_instance(self) -> ModelInstance:
        """ The median of the probability density function (PDF) of every parameter marginalized in 1D, returned
        as a model instance."""
        return self.model.instance_from_vector(vector=self.median_pdf_vector)

    def vector_at_sigma(self, sigma) -> [(float, float)]:
        """ The value of every parameter marginalized in 1D at an input sigma value of its probability density function
        (PDF), returned as two lists of values corresponding to the lower and upper values parameter values.

        For example, if sigma is 1.0, the marginalized values of every parameter at 31.7% and 68.2% percentiles of each
        PDF is returned.

        This does not account for covariance between parameters. For example, if two parameters (x, y) are degenerate
        whereby x decreases as y gets larger to give the same PDF, this function will still return both at their
        upper values. Thus, caution is advised when using the function to reperform a model-fits.

        For *Dynesty*, this is estimated using *corner.py* if the samples have converged, by sampling the density
        function at an input PDF %. If not converged, a crude estimate using the range of values of the current
        physical live points is used.

        Parameters
        ----------
        sigma : float
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF)."""

        if self.pdf_converged:
            limit = math.erf(0.5 * sigma * math.sqrt(2))

            lower_errors = [
                quantile(x=params, q=1.0 - limit, weights=self.weights)[0]
                for params in self.parameters_extract
            ]

            upper_errors = [
                quantile(x=params, q=limit, weights=self.weights)[0]
                for params in self.parameters_extract
            ]

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

    def error_vector_at_sigma(self, sigma) -> [(float, float)]:
        """The lower and upper error of every parameter marginalized in 1D at an input sigma value of its probability
        density function (PDF), returned as a list.

        See *vector_at_sigma* for a full description of how the parameters at sigma are computed.

        Parameters
        -----------
        sigma : float
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the \
            PDF).
        """
        error_vector_lower = self.error_vector_at_lower_sigma(sigma=sigma)
        error_vector_upper = self.error_vector_at_upper_sigma(sigma=sigma)
        return [(lower, upper) for lower, upper in zip(error_vector_lower, error_vector_upper)]

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

    def error_magnitude_vector_at_sigma(self, sigma) -> [float]:
        """ The magnitude of every error after marginalization in 1D at an input sigma value of the probability density
        function (PDF), returned as two lists of values corresponding to the lower and upper errors.

        For example, if sigma is 1.0, the difference in the inferred values marginalized at 31.7% and 68.2% percentiles
        of each PDF is returned.

        Parameters
        ----------
        sigma : float
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF)."""
        uppers = self.vector_at_upper_sigma(sigma=sigma)
        lowers = self.vector_at_lower_sigma(sigma=sigma)
        return list(map(lambda upper, lower: upper - lower, uppers, lowers))

    def error_instance_at_sigma(self, sigma) -> ModelInstance:
        """ The error of every parameter marginalized in 1D at an input sigma value of its probability density function
        (PDF), returned as a list of model instances corresponding to the lower and upper errors.

        See *vector_at_sigma* for a full description of how the parameters at sigma are computed.

        Parameters
        ----------
        sigma : float
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF)."""
        return self.model.instance_from_vector(
            vector=self.error_magnitude_vector_at_sigma(sigma=sigma),
            assert_priors_in_limits=False,
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
            vector=self.error_vector_at_upper_sigma(sigma=sigma),
            assert_priors_in_limits=False,
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
            vector=self.error_vector_at_lower_sigma(sigma=sigma),
            assert_priors_in_limits=False,
        )

    def gaussian_priors_at_sigma(self, sigma) -> [list]:
        """`GaussianPrior`s of every parameter used to link its inferred values and errors to priors used to sample the
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
        of the `NonLinearSearch` as values offset from the 'true' model. For example, a value 0.0 means the non-linear
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

        This uses *corner.py* to plot:

         - The marginalize 1D PDF of every parameter.
         - The marginalized 2D PDF of every parameter pair.
         - A Triangle plot of the 2D and 1D PDF's.
         """

        pass

        # import getdist.plots
        # import matplotlib
        #
        # backend = conf.instance["visualize"]["general"]["general"]["backend"]
        # if not backend in "default":
        #     matplotlib.use(backend)
        # if conf.instance["general"]["hpc"]["hpc_mode"]:
        #     matplotlib.use("Agg")
        # import matplotlib.pyplot as plt
        #
        # pdf_plot = getdist.plots.corner.pyPlotter()
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
        # plot_pdf_triangle = conf.instance["visualize"]["plots"]["pdf"]["triangle"]
        #
        # if plot_pdf_triangle:
        #
        #     try:
        #         pdf_plot.triangle_plot(roots=self.getdist_samples)
        #         pdf_plot.export(fname="{}/pdf_triangle.png".format(self.paths.pdf_path))
        #     except Exception as e:
        #         logger.exception(e)
        #         print(
        #             "The PDF triangle of this `NonLinearSearch` could not be plotted. This is most likely due to a "
        #             "lack of smoothness in the sampling of parameter space. Sampler further by decreasing the "
        #             "parameter evidence_tolerance."
        #         )
        #
        # plt.close()


class MCMCSamples(PDFSamples):
    def __init__(
            self,
            model: ModelMapper,
            samples: List[Sample],
            auto_correlation_times: np.ndarray,
            auto_correlation_check_size: int,
            auto_correlation_required_length: int,
            auto_correlation_change_threshold: float,
            total_walkers: int,
            total_steps: int,
            unconverged_sample_size: int = 100,
            time: float = None,
    ):
        """
        Attributes
        ----------
        total_walkers : int
            The total number of walkers used by this MCMC non-linear search.
        total_steps : int
            The total number of steps taken by each walker of this MCMC `NonLinearSearch` (the total samples is equal
            to the total steps * total walkers).
        """

        super().__init__(
            model=model,
            samples=samples,
            unconverged_sample_size=unconverged_sample_size,
            time=time,
        )

        self.total_walkers = total_walkers
        self.total_steps = total_steps
        self.auto_correlation_times = auto_correlation_times
        self.auto_correlation_check_size = auto_correlation_check_size
        self.auto_correlation_required_length = auto_correlation_required_length
        self.auto_correlation_change_threshold = auto_correlation_change_threshold
        self.log_evidence = None

    @classmethod
    def from_table(self, filename: str, model, number_live_points=None):
        """
        Write a table of parameters, posteriors, priors and likelihoods

        Parameters
        ----------
        filename
            Where the table is to be written
        """

        samples = load_from_table(filename=filename)

        return OptimizerSamples(
            model=model,
            samples=samples
        )

    def info_to_json(self, filename):

        info = {
            "auto_correlation_times": None,
            "auto_correlation_check_size": self.auto_correlation_check_size,
            "auto_correlation_required_length": self.auto_correlation_required_length,
            "auto_correlation_change_threshold": self.auto_correlation_change_threshold,
            "total_walkers": self.total_walkers,
            "total_steps": self.total_steps,
            "time": self.time,
        }

        with open(filename, 'w') as outfile:
            json.dump(info, outfile)

    @property
    def pdf_converged(self):
        """ To analyse and visualize samples using *corner.py*, the analysis must be sufficiently converged to produce
        smooth enough PDF for analysis. This property checks whether the non-linear search's samples are sufficiently
        converged for *corner.py* use.

        Emcee samples can be analysed by corner.py irrespective of how long the sampler has run, albeit low run times
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
        raise NotImplementedError()

    @property
    def previous_auto_correlation_times(self) -> [float]:
        raise NotImplementedError()

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
            samples: List[Sample],
            number_live_points: int,
            log_evidence: float,
            total_samples: float,
            unconverged_sample_size: int = 100,
            time: float = None,
    ):
        """The *Output* classes in **PyAutoFit** provide an interface between the results of a `NonLinearSearch` (e.g.
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
            samples=samples,
            unconverged_sample_size=unconverged_sample_size,
            time=time,
        )

        self.number_live_points = number_live_points
        self.log_evidence = log_evidence
        self._total_samples = total_samples

    @property
    def total_samples(self):
        return self._total_samples

    def info_to_json(self, filename):
        info = {
            "log_evidence": self.log_evidence,
            "total_samples": self.total_samples,
            "unconverged_sample_size": self.unconverged_sample_size,
            "time": self.time,
            "number_live_points": self.number_live_points
        }

        with open(filename, 'w') as outfile:
            json.dump(info, outfile)

    @property
    def total_accepted_samples(self) -> int:
        """The total number of accepted samples performed by the nested sampler.
        """
        return len(self.log_likelihoods)

    @property
    def acceptance_ratio(self) -> float:
        """The ratio of accepted samples to total samples."""
        return self.total_accepted_samples / self.total_samples

    def samples_within_parameter_range(
            self,
            parameter_index : int,
            parameter_range : [float, float]
    ) -> "NestSamples":
        """
        Returns a new set of Samples where all points without parameter values inside a specified range removed.

        For example, if our `Samples` object was for a model with 4 parameters are consistent of the following 3 sets
        of parameters:

        [[1.0, 2.0, 3.0, 4.0]]
        [[100.0, 2.0, 3.0, 4.0]]
        [[1.0, 2.0, 3.0, 4.0]]

        This function for `parameter_index=0` and `parameter_range=[0.0, 99.0]` would remove the second sample because
        the value 100.0 is outside the range 0.0 -> 99.0.

        Parameters
        ----------
        parameter_index : int
            The 1D index of the parameter (in the model's vector representation) whose values lie between the parameter
            range if a sample is kept.
        parameter_range : [float, float]
            The minimum and maximum values of the range of parameter values this parameter must lie between for it
            to be kept.
        """

        parameter_list = []
        log_likelihoods = []
        log_priors = []
        weights = []

        for sample in self.samples:

            parameters = sample.parameters_for_model(model=self.model)

            if (parameters[parameter_index] > parameter_range[0]) and (parameters[parameter_index] < parameter_range[1]):

                parameter_list.append(parameters)
                log_likelihoods.append(sample.log_likelihood)
                log_priors.append(sample.log_prior)
                weights.append(sample.weights)

        samples = Sample.from_lists(
            model=self.model,
            parameters=parameter_list,
            log_likelihoods=log_likelihoods,
            log_priors=log_priors,
            weights=weights
        )

        return NestSamples(
            model=self.model,
            samples=samples,
            number_live_points=self.number_live_points,
            log_evidence=self.log_evidence,
            total_samples=self.total_samples,
            unconverged_sample_size=self.unconverged_sample_size,
            time=self.time
        )

def quantile(x, q, weights=None):
    """
    Copied from corner.py

    Compute sample quantiles with support for weighted samples.
    Note
    ----
    When ``weights`` is ``None``, this method simply calls numpy's percentile
    function with the values of ``q`` multiplied by 100.
    Parameters
    ----------
    x : array_like[nsamples,]
       The samples.
    q : array_like[nquantiles,]
       The list of quantiles to compute. These should all be in the range
       ``[0, 1]``.
    weights : Optional[array_like[nsamples,]]
        An optional weight corresponding to each sample. These
    Returns
    -------
    quantiles : array_like[nquantiles,]
        The sample quantiles computed at ``q``.
    Raises
    ------
    ValueError
        For invalid quantiles; ``q`` not in ``[0, 1]`` or dimension mismatch
        between ``x`` and ``weights``.
    """
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0 and 1")

    if weights is None:
        return np.percentile(x, list(100.0 * q))
    else:
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x)")
        idx = np.argsort(x)
        sw = weights[idx]
        cdf = np.cumsum(sw)[:-1]
        cdf /= cdf[-1]
        cdf = np.append(0, cdf)
        return np.interp(q, cdf, x[idx]).tolist()
