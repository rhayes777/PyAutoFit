import csv
import json
import math
from copy import copy
from typing import List, Optional

import numpy as np

from autofit.mapper.model import ModelInstance
from autofit.mapper.model_mapper import ModelMapper
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.mcmc.auto_correlations import AutoCorrelations, AutoCorrelationsSettings


class Sample:
    def __init__(
            self,
            log_likelihood: float,
            log_prior: float,
            weight: float,
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
        weight
        kwargs
            Dictionary mapping model paths to values for the sample
        """
        self.log_likelihood = log_likelihood
        self.log_prior = log_prior
        self.weight = weight
        self.kwargs = kwargs

    @property
    def log_posterior(self) -> float:
        """
        Compute the posterior
        """
        return self.log_likelihood + self.log_prior

    def parameter_lists_for_model(
            self,
            model: AbstractPriorModel,
            paths=None
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

    @classmethod
    def from_lists(
            cls,
            model: AbstractPriorModel,
            parameter_lists: List[List[float]],
            log_likelihood_list: List[float],
            log_prior_list: List[float],
            weight_list: List[float]
    ) -> List["Sample"]:
        """
        Convenience method to create a list of samples
        from lists of contained values

        Parameters
        ----------
        model
        parameter_lists
        log_likelihood_list
        log_prior_list
        weight_list

        Returns
        -------
        A list of samples
        """
        samples = list()

        # Another speed up.
        model_component_and_parameter_names = model.model_component_and_parameter_names

        for params, log_likelihood, log_prior, weight in zip(
                parameter_lists,
                log_likelihood_list,
                log_prior_list,
                weight_list
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
                    weight=weight,
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
                self.parameter_lists_for_model(model)
            )
        except KeyError:
            paths = model.model_component_and_parameter_names
            return model.instance_from_vector(
                self.parameter_lists_for_model(model, paths)
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
            model: AbstractPriorModel,
            time: Optional[float] = None,
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
        self.time = time

    @property
    def samples(self):
        raise NotImplementedError

    @property
    def parameter_lists(self):

        paths = self.model.model_component_and_parameter_names

        return [
            sample.parameter_lists_for_model(
                self.model, paths
            )
            for sample in self.samples
        ]

    @property
    def total_samples(self):
        return len(self.samples)

    @property
    def weight_list(self):
        return [
            sample.weight
            for sample
            in self.samples
        ]

    @property
    def log_likelihood_list(self):
        return [
            sample.log_likelihood
            for sample
            in self.samples
        ]

    @property
    def log_posterior_list(self):
        return [
            sample.log_posterior
            for sample
            in self.samples
        ]

    @property
    def log_prior_list(self):
        return [
            sample.log_prior
            for sample
            in self.samples
        ]

    @property
    def parameters_extract(self):
        return [
            [params[i] for params in self.parameter_lists]
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
            "weight",
        ]

    @property
    def _rows(self) -> List[List[float]]:
        """
        Rows in the samples table
        """

        log_likelihood_list = self.log_likelihood_list
        log_prior_list = self.log_prior_list
        log_posterior_list = self.log_posterior_list
        weight_list = self.weight_list

        for index, row in enumerate(self.parameter_lists):
            yield row + [
                log_likelihood_list[index],
                log_prior_list[index],
                log_posterior_list[index],
                weight_list[index],
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

    @property
    def info_json(self):
        return {}

    def info_to_json(self, filename):
        with open(filename, 'w') as outfile:
            json.dump(self.info_json, outfile)

    @property
    def max_log_likelihood_sample(self) -> Sample:
        """The index of the sample with the highest log likelihood."""
        most_likely_sample = None
        for sample in self.samples:
            if most_likely_sample is None or sample.log_likelihood > most_likely_sample.log_likelihood:
                most_likely_sample = sample
        return most_likely_sample

    @property
    def max_log_posterior_sample(self) -> Sample:
        return self.samples[
            self.max_log_posterior_index
        ]

    @property
    def max_log_likelihood_vector(self) -> [float]:
        """ The parameters of the maximum log likelihood sample of the `NonLinearSearch` returned as a list of values."""
        return self.max_log_likelihood_sample.parameter_lists_for_model(
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
        return int(np.argmax(self.log_posterior_list))

    @property
    def max_log_posterior_vector(self) -> [float]:
        """ The parameters of the maximum log posterior sample of the `NonLinearSearch` returned as a list of values."""
        return self.parameter_lists[self.max_log_posterior_index]

    @property
    def max_log_posterior_instance(self) -> ModelInstance:
        """  The parameters of the maximum log posterior sample of the `NonLinearSearch` returned as a model instance."""
        return self.model.instance_from_vector(vector=self.max_log_posterior_vector)

    def gaussian_priors_at_sigma(self, sigma) -> [list]:
        """`GaussianPrior`s of every parameter used to link its inferred values and errors to priors used to sample the
        same (or similar) parameters in a subsequent search, where:

         - The mean is given by maximum log likelihood model values.
         - Their errors are omitted, as this information is not available from an search. When these priors are
           used to link to another search, it will thus automatically use the prior config values.

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
        return self.model.instance_from_vector(vector=self.parameter_lists[sample_index])

    def minimise(self) -> "OptimizerSamples":
        """
        A copy of this object with only important samples retained
        """
        samples = copy(self)
        samples.model = None
        samples._samples = list({
            self.max_log_likelihood_sample,
            self.max_log_posterior_sample
        })
        return samples


class PDFSamples(OptimizerSamples):
    def __init__(
            self,
            model: AbstractPriorModel,
            unconverged_sample_size: int = 100,
            time: Optional[float] = None,
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
            time=time,
        )

        self._unconverged_sample_size = int(unconverged_sample_size)

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

        return StoredSamples(
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
        if np.max(self.weight_list) > 0.99:
            return False
        return True

    @property
    def median_pdf_vector(self) -> [float]:
        """ The median of the probability density function (PDF) of every parameter marginalized in 1D, returned
        as a list of values."""
        if self.pdf_converged:
            return [
                quantile(x=params, q=0.5, weights=self.weight_list)[0]
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
                quantile(x=params, q=1.0 - limit, weights=self.weight_list)[0]
                for params in self.parameters_extract
            ]

            upper_errors = [
                quantile(x=params, q=limit, weights=self.weight_list)[0]
                for params in self.parameters_extract
            ]

            return [(lower, upper) for lower, upper in zip(lower_errors, upper_errors)]

        parameters_min = list(
            np.min(self.parameter_lists[-self.unconverged_sample_size:], axis=0)
        )
        parameters_max = list(
            np.max(self.parameter_lists[-self.unconverged_sample_size:], axis=0)
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
        same (or similar) parameters in a subsequent search, where:

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


class MCMCSamples(PDFSamples):
    def __init__(
            self,
            model: ModelMapper,
            auto_correlation_settings: AutoCorrelationsSettings,
            unconverged_sample_size: int = 100,
            time: Optional[float] = None,
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

        self.auto_correlation_settings = auto_correlation_settings

        super().__init__(
            model=model,
            unconverged_sample_size=unconverged_sample_size,
            time=time,
        )

    @property
    def total_walkers(self):
        raise NotImplementedError

    @property
    def total_steps(self):
        raise NotImplementedError

    @property
    def auto_correlations(self):
        raise NotImplementedError

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

    @property
    def info_json(self):
        return {
            "times": None,
            "check_size": self.auto_correlations.check_size,
            "required_length": self.auto_correlations.required_length,
            "change_threshold": self.auto_correlations.change_threshold,
            "total_walkers": self.total_walkers,
            "total_steps": self.total_steps,
            "time": self.time,
        }

    @property
    def pdf_converged(self):
        """ To analyse and visualize samples using *corner.py*, the analysis must be sufficiently converged to produce
        smooth enough PDF for analysis. This property checks whether the non-linear search's samples are sufficiently
        converged for *corner.py* use.

        Emcee samples can be analysed by corner.py irrespective of how long the sampler has run, albeit low run times
        will likely produce inaccurate results."""
        try:
            samples_after_burn_in = self.samples_after_burn_in
            if len(samples_after_burn_in) == 0:
                return False
            return True
        except ValueError:
            return False

    @property
    def samples_after_burn_in(self) -> [list]:
        """The emcee samples with the initial burn-in samples removed.

        The burn-in period is estimated using the auto-correlation times of the parameters."""
        raise NotImplementedError()

    @property
    def converged(self) -> bool:
        """
        Whether the emcee samples have converged on a solution or if they are still in a burn-in period, based on the
        auto correlation times of parameters.
        """
        return self.auto_correlations.check_if_converged(
            total_samples=self.total_samples
        )

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
            np.min(self.parameter_lists[-self.unconverged_sample_size:], axis=0)
        )
        parameters_max = list(
            np.max(self.parameter_lists[-self.unconverged_sample_size:], axis=0)
        )

        return [
            (parameters_min[index], parameters_max[index])
            for index in range(len(parameters_min))
        ]

    @property
    def log_evidence(self):
        return None


class NestSamples(PDFSamples):
    def __init__(
            self,
            model: AbstractPriorModel,
            unconverged_sample_size: int = 100,
            time: Optional[float] = None,
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
            unconverged_sample_size=unconverged_sample_size,
            time=time,
        )

    @property
    def number_live_points(self):
        raise NotImplementedError

    @property
    def log_evidence(self):
        raise NotImplementedError

    @property
    def total_samples(self):
        raise NotImplementedError

    @property
    def info_json(self):
        return {
            "log_evidence": self.log_evidence,
            "total_samples": self.total_samples,
            "unconverged_sample_size": self.unconverged_sample_size,
            "time": self.time,
            "number_live_points": self.number_live_points
        }

    @property
    def total_accepted_samples(self) -> int:
        """The total number of accepted samples performed by the nested sampler.
        """
        return len(self.log_likelihood_list)

    @property
    def acceptance_ratio(self) -> float:
        """The ratio of accepted samples to total samples."""
        return self.total_accepted_samples / self.total_samples

    def samples_within_parameter_range(
            self,
            parameter_index: int,
            parameter_range: [float, float]
    ) -> "StoredSamples":
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
        log_likelihood_list = []
        log_prior_list = []
        weight_list = []

        for sample in self.samples:

            parameters = sample.parameter_lists_for_model(model=self.model)

            if (parameters[parameter_index] > parameter_range[0]) and (
                    parameters[parameter_index] < parameter_range[1]):

                parameter_list.append(parameters)
                log_likelihood_list.append(sample.log_likelihood)
                log_prior_list.append(sample.log_prior)
                weight_list.append(sample.weight)

        samples = Sample.from_lists(
            model=self.model,
            parameter_lists=parameter_list,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list
        )

        return StoredSamples(
            model=self.model,
            samples=samples,
            unconverged_sample_size=self.unconverged_sample_size,
        )


class StoredSamples(PDFSamples):

    def __init__(
            self,
            model: AbstractPriorModel,
            samples,
            unconverged_sample_size: int = 100,
            time: Optional[float] = None,
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
            time=time,
        )

        self._samples = samples
        self._unconverged_sample_size = int(unconverged_sample_size)

    @property
    def samples(self):
        return self._samples

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
        between ``x`` and ``weight_list``.
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
