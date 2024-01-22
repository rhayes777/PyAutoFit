import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from autoconf import conf
from autofit.mapper.model import ModelInstance
from autofit.mapper.prior_model.abstract import AbstractPriorModel, Path
from autofit.non_linear.samples.sample import Sample, load_from_table
from autofit.non_linear.samples.samples import (
    to_instance,
    to_instance_sigma,
    to_instance_input,
)
from .samples import Samples
from .summary import SamplesSummary


class SamplesPDF(Samples):
    def __init__(
        self,
        model: AbstractPriorModel,
        sample_list: List[Sample],
        samples_info: Optional[Dict] = None,
        search_internal: Optional = None,
    ):
        """
        The `Samples` classes in **PyAutoFit** provide an interface between the search_internal of
        a `NonLinearSearch` (e.g. as files on your hard-disk) and Python.

        For example, the output class can be used to load an instance of the best-fit model, get an instance of any
        individual sample by the `NonLinearSearch` and return information on the likelihoods, errors, etc.

        This class stores samples of searches which provide the probability distribution function (PDF) of the
        model fit (e.g. nested samplers, MCMC).

        To use a library's in-built visualization tools results are optionally stored in their native internal format
        using the `search_internal` attribute.

        Parameters
        ----------
        model
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        sample_list
            The list of `Samples` which contains the paramoeters, likelihood, weights, etc. of every sample taken
            by the non-linear search.
        samples_info
            Contains information on the samples (e.g. total iterations, time to run the search, etc.).
        search_internal
            The nested sampler's results in their native internal format for interfacing its visualization library.
        """

        super().__init__(
            model=model,
            sample_list=sample_list,
            samples_info=samples_info,
            search_internal=search_internal,
        )

    def summary(self):
        try:
            covariance_matrix = self.covariance_matrix
        except Exception as e:
            logging.warning(f"Could not create covariance matrix: {e}")
            covariance_matrix = None
        return SamplesSummary(
            max_log_likelihood_sample=self.max_log_likelihood_sample,
            model=self.model,
            covariance_matrix=covariance_matrix,
            log_evidence=self.log_evidence,
        )

    @classmethod
    def from_table(cls, filename: str, model, number_live_points=None):
        """
        Write a table of parameters, posteriors, priors and likelihoods

        Parameters
        ----------
        filename
            Where the table is to be written
        """
        from .stored import SamplesStored

        sample_list = load_from_table(filename=filename)

        return SamplesStored(model=model, sample_list=sample_list)

    @property
    def unconverged_sample_size(self):
        """
        If a set of samples are unconverged, alternative methods to compute their means, errors, etc are used as
        an alternative to corner.py.

        These use a subset of samples spanning the range from the most recent sample to the valaue of the
        unconverted_sample_size. However, if there are fewer samples than this size, we change the size to be the
        the size of the total number of samples.
        """

        unconverged_sample_size = conf.instance["general"]["output"][
            "unconverged_sample_size"
        ]

        if self.total_samples > unconverged_sample_size:
            return unconverged_sample_size
        return self.total_samples

    @property
    def pdf_converged(self) -> bool:
        """
        To analyse and visualize samples the analysis must be sufficiently converged to produce smooth enough
        PDF for error estimate and PDF generation.

        This property checks whether the non-linear search's samples are sufficiently converged for this, by checking
        if one sample's weight contains > 99% of the weight. If this is the case, it implies the convergence necessary
        for error estimate and visualization has not been met.

        This does not necessarily imply the `NonLinearSearch` has converged overall, only that errors and visualization
        can be performed numerically.
        """
        if np.max(self.weight_list) > 0.99:
            return False
        return True

    @to_instance
    def median_pdf(self, as_instance: bool = True) -> List[float]:
        """
        The median of the probability density function (PDF) of every parameter marginalized in 1D, returned
        as a model instance or list of values.
        """
        if self.pdf_converged:
            return [
                quantile(x=params, q=0.5, weights=self.weight_list)[0]
                for params in self.parameters_extract
            ]
        return self.max_log_likelihood(as_instance=False)

    @to_instance_sigma
    def values_at_sigma(
        self, sigma: float, as_instance: bool = True
    ) -> [Tuple, ModelInstance]:
        """
        The value of every parameter marginalized in 1D at an input sigma value of its probability density function
        (PDF), returned as two lists of values corresponding to the lower and upper values parameter values.

        For example, if sigma is 1.0, the marginalized values of every parameter at 31.7% and 68.2% percentiles of each
        PDF is returned.

        This does not account for covariance between parameters. For example, if two parameters (x, y) are degenerate
        whereby x decreases as y gets larger to give the same PDF, this function will still return both at their
        upper values. Thus, caution is advised when using the function to reperform a model-fits.

        For Dynesty, this is estimated using corner.py if the samples have converged, by sampling the density
        function at an input PDF %. If not converged, a crude estimate using the range of values of the current
        physical live points is used.

        Parameters
        ----------
        sigma
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF).
        """

        if self.pdf_converged:
            low_limit = (1 - math.erf(sigma / math.sqrt(2))) / 2

            lower_errors = [
                quantile(x=params, q=low_limit, weights=self.weight_list)[0]
                for params in self.parameters_extract
            ]
            upper_errors = [
                quantile(x=params, q=1 - low_limit, weights=self.weight_list)[0]
                for params in self.parameters_extract
            ]

            return [(lower, upper) for lower, upper in zip(lower_errors, upper_errors)]

        parameters_min = list(
            np.min(self.parameter_lists[-self.unconverged_sample_size :], axis=0)
        )
        parameters_max = list(
            np.max(self.parameter_lists[-self.unconverged_sample_size :], axis=0)
        )

        return [
            (parameters_min[index], parameters_max[index])
            for index in range(len(parameters_min))
        ]

    @to_instance_sigma
    def values_at_upper_sigma(
        self, sigma: float, as_instance: bool = True
    ) -> Union[List, ModelInstance]:
        """
        The upper value of every parameter marginalized in 1D at an input sigma value of its probability density
        function (PDF), returned as a list.

        See values_at_sigma for a full description of how the parameters at sigma are computed.

        Parameters
        ----------
        sigma
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF).
        """
        return list(
            map(lambda param: param[1], self.values_at_sigma(sigma, as_instance=False))
        )

    @to_instance_sigma
    def values_at_lower_sigma(
        self, sigma: float, as_instance: bool = True
    ) -> Union[List, ModelInstance]:
        """
        The lower value of every parameter marginalized in 1D at an input sigma value of its probability density
        function (PDF), returned as a list.

        See values_at_sigma for a full description of how the parameters at sigma are computed.

        Parameters
        ----------
        sigma
            The sigma limit within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF).
        """
        return list(
            map(lambda param: param[0], self.values_at_sigma(sigma, as_instance=False))
        )

    @to_instance_sigma
    def errors_at_sigma(
        self, sigma: float, as_instance: bool = True
    ) -> [Tuple, ModelInstance]:
        """
        The lower and upper error of every parameter marginalized in 1D at an input sigma value of its probability
        density function (PDF), returned as a list.

        See values_at_sigma for a full description of how the parameters at sigma are computed.

        Parameters
        ----------
        sigma
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF).
        """
        error_vector_lower = self.errors_at_lower_sigma(sigma=sigma, as_instance=False)
        error_vector_upper = self.errors_at_upper_sigma(sigma=sigma, as_instance=False)
        return [
            (lower, upper)
            for lower, upper in zip(error_vector_lower, error_vector_upper)
        ]

    @to_instance_sigma
    def errors_at_upper_sigma(
        self, sigma: float, as_instance: bool = True
    ) -> Union[List, ModelInstance]:
        """
        The upper error of every parameter marginalized in 1D at an input sigma value of its probability density
        function (PDF), returned as a list.

        See values_at_sigma for a full description of how the parameters at sigma are computed.

        Parameters
        ----------
        sigma
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF).
        """
        uppers = self.values_at_upper_sigma(sigma=sigma, as_instance=False)
        return list(
            map(
                lambda upper, median_pdf: upper - median_pdf,
                uppers,
                self.median_pdf(as_instance=False),
            )
        )

    @to_instance_sigma
    def errors_at_lower_sigma(
        self, sigma: float, as_instance: bool = True
    ) -> Union[List, ModelInstance]:
        """
        The lower error of every parameter marginalized in 1D at an input sigma value of its probability density
        function (PDF), returned as a list.

        See values_at_sigma for a full description of how the parameters at sigma are computed.

        Parameters
        ----------
        sigma
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF).
        """
        lowers = self.values_at_lower_sigma(sigma=sigma, as_instance=False)
        return list(
            map(
                lambda lower, median_pdf: median_pdf - lower,
                lowers,
                self.median_pdf(as_instance=False),
            )
        )

    @to_instance_sigma
    def error_magnitudes_at_sigma(
        self, sigma: float, as_instance: bool = True
    ) -> Union[List, ModelInstance]:
        """
        The magnitude of every error after marginalization in 1D at an input sigma value of the probability density
        function (PDF), returned as two lists of values corresponding to the lower and upper errors.

        For example, if sigma is 1.0, the difference in the inferred values marginalized at 31.7% and 68.2% percentiles
        of each PDF is returned.

        Parameters
        ----------
        sigma
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF).
        """
        uppers = self.values_at_upper_sigma(sigma=sigma, as_instance=False)
        lowers = self.values_at_lower_sigma(sigma=sigma, as_instance=False)
        return list(map(lambda upper, lower: upper - lower, uppers, lowers))

    def gaussian_priors_at_sigma(self, sigma: float) -> [List]:
        """
        `GaussianPrior`s of every parameter used to link its inferred values and errors to priors used to sample the
        same (or similar) parameters in a subsequent search, where:

        - The mean is given by their most-probable values (using median_pdf(as_instance=False)).
        - Their errors are computed at an input sigma value (using errors_at_sigma).

        Parameters
        ----------
        sigma
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF).
        """

        means = self.median_pdf(as_instance=False)

        uppers = self.values_at_upper_sigma(sigma=sigma, as_instance=False)
        lowers = self.values_at_lower_sigma(sigma=sigma, as_instance=False)

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

    @to_instance
    def draw_randomly_via_pdf(
        self, as_instance: bool = True
    ) -> Union[List, ModelInstance]:
        """
        The parameter vector of an individual sample of the non-linear search drawn randomly from the PDF, returned as
        a 1D list.

        The draw is weighted by the sample weights to ensure that the sample is drawn from the PDF (which is important
        for non-linear searches like nested sampling).
        """

        sample_index = np.random.choice(
            a=range(len(self.sample_list)), p=self.weight_list
        )

        return self.parameter_lists[sample_index][:]

    @to_instance_input
    def offset_values_via_input_values(
        self, input_vector: List, as_instance: bool = True
    ) -> Union[List, ModelInstance]:
        """
        The values of an input_vector offset by the median_pdf(as_instance=False) (the PDF medians).

        If the 'true' values of a model are known and input as the input_vector, this function returns the results
        of the `NonLinearSearch` as values offset from the 'true' model. For example, a value 0.0 means the non-linear
        search estimated the model parameter value correctly.

        Parameters
        ----------
        input_vector
            A 1D list of values the most-probable model is offset by. This list must have the same dimensions as free
            parameters in the model-fit.
        """
        return list(
            map(
                lambda input, median_pdf: median_pdf - input,
                input_vector,
                self.median_pdf(as_instance=False),
            )
        )

    @property
    def covariance_matrix(self) -> np.ndarray:
        """
        Compute the covariance matrix of the non-linear search samples, using the method `np.cov()` which is described
        at the following link:

        https://numpy.org/doc/stable/reference/generated/numpy.cov.html

        Follow that link for a description of what the covariance matrix is.

        Returns
        -------
        ndarray
            A covariance matrix of shape [total_parameters, total_parameters] for the model parameters of the
            non-linear search.
        """
        if len(self.parameter_lists) == 1:
            return np.eye(1)
        return np.cov(m=self.parameter_lists, rowvar=False, aweights=self.weight_list)

    def save_covariance_matrix(self, filename: Union[Path, str]):
        """
        Save the covariance matrix as a CSV file.

        Parameters
        ----------
        filename
            The filename the covariance matrix is saved to.
        """
        # noinspection PyTypeChecker
        np.savetxt(filename, self.covariance_matrix, delimiter=",")

    @property
    def log_evidence(self):
        return None


def marginalize(
    parameter_list: List, sigma: float, weight_list: Optional[List] = None
) -> Tuple[float, float, float]:
    low_limit = (1 - math.erf(sigma / math.sqrt(2))) / 2

    median = quantile(x=parameter_list, q=0.5, weights=weight_list)[0]
    lower = quantile(x=parameter_list, q=low_limit, weights=weight_list)[0]
    upper = quantile(x=parameter_list, q=1 - low_limit, weights=weight_list)[0]

    return median, lower, upper


def quantile(x, q, weights=None):
    """
    Copied from corner.py

    Compute sample quantiles with support for weighted samples.

    Note
    ----
    When ``weight_list`` is ``None``, this method simply calls numpy's percentile
    function with the values of ``q`` multiplied by 100.

    Parameters
    ----------
    x : array_like[nsamples,]
       The samples.
    q : array_like[nquantiles,]
       The list of quantiles to compute. These should all be in the range
       ``[0, 1]``.
    weights: Optional[array_like[nsamples,]]
        An optional weight corresponding to each sample. These

    Returns
    -------
    quantiles : array_like[nquantiles,]
        The sample quantiles computed at ``q``.

    Raises
    ------
    ValueError
        For invalid quantiles; ``q`` not in ``[0, 1]`` or dimension mismatch between ``x`` and ``weight_list``.
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
            raise ValueError("Dimension mismatch: len(weight_list) != len(x)")
        idx = np.argsort(x)
        sw = weights[idx]
        cdf = np.cumsum(sw)[:-1]
        cdf /= cdf[-1]
        cdf = np.append(0, cdf)
        return np.interp(q, cdf, x[idx]).tolist()
