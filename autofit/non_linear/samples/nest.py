from typing import List, Optional
import warnings

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.samples.pdf import PDFSamples
from autofit.non_linear.samples.samples import Sample
from autofit.non_linear.samples.stored import StoredSamples

from autofit import exc

class NestSamples(PDFSamples):

    def __init__(
            self,
            model: AbstractPriorModel,
            sample_list: List[Sample],
            number_live_points: Optional[int] = None,
            unconverged_sample_size: int = 100,
            time: Optional[float] = None,
            results_internal: Optional = None,
    ):
        """
        The `Samples` classes in **PyAutoFit** provide an interface between the results_internal of
        a `NonLinearSearch` (e.g. as files on your hard-disk) and Python.

        For example, the output class can be used to load an instance of the best-fit model, get an instance of any
        individual sample by the `NonLinearSearch` and return information on the likelihoods, errors, etc.

        This class stores the samples of nested sampler model-fit (e.g. `dynesty`, `UltraNest`). To use a library's
        in-built visualization tools results are optionally stored in their native internal format using the
        `results_internal` attribute.

        Parameters
        ----------
        model
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        sample_list
            The list of `Samples` which contains the paramoeters, likelihood, weights, etc. of every sample taken
            by the non-linear search.
        number_live_points
            The number of live points used by the nested sampler.
        unconverged_sample_size
            If the samples are for a search that is yet to convergence, a reduced set of samples are used to provide
            a rough estimate of the parameters. The number of samples is set by this parameter.
        time
            The time taken to perform the model-fit, which is passed around `Samples` objects for outputting
            information on the overall fit.
        results_internal
            The nested sampler's results in their native internal format for interfacing its visualization library.
        """

        self._number_live_points = number_live_points

        super().__init__(
            model=model,
            sample_list=sample_list,
            unconverged_sample_size=unconverged_sample_size,
            time=time,
            results_internal=results_internal
        )

    def __add__(
            self,
            other: "NestSamples"
    ) -> "NestSamples":
        """
        Samples can be added together, which combines their `sample_list` meaning that inferred parameters are
        computed via their joint PDF.

        For UltraNest samples there are no tools for combining results in their native format, therefore these
        `results_internal` are set to None and support for visualization is disabled.

        Parameters
        ----------
        other
            Another Samples class

        Returns
        -------
        A class that combined the samples of the two Samples objects.
        """

        self._check_addition(other=other)

        warnings.warn(
            f"Addition of {self.__class__.__name__} cannot retain results in native format. "
            "Visualization of summed samples diabled.",
            exc.SamplesWarning
        )

        return self.__class__(
            model=self.model,
            sample_list=self.sample_list + other.sample_list,
            number_live_points=self._number_live_points,
            unconverged_sample_size=self.unconverged_sample_size,
            time=self.time,
            results_internal=None
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
        """
        The total number of accepted samples performed by the nested sampler.
        """
        return len(self.log_likelihood_list)

    @property
    def acceptance_ratio(self) -> float:
        """
        The ratio of accepted samples to total samples.
        """
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
        parameter_index
            The 1D index of the parameter (in the model's vector representation) whose values lie between the parameter
            range if a sample is kept.
        parameter_range
            The minimum and maximum values of the range of parameter values this parameter must lie between for it
            to be kept.
        """

        parameter_list = []
        log_likelihood_list = []
        log_prior_list = []
        weight_list = []

        for sample in self.sample_list:

            parameters = sample.parameter_lists_for_model(model=self.model)

            if (parameters[parameter_index] > parameter_range[0]) and (
                    parameters[parameter_index] < parameter_range[1]):

                parameter_list.append(parameters)
                log_likelihood_list.append(sample.log_likelihood)
                log_prior_list.append(sample.log_prior)
                weight_list.append(sample.weight)

        sample_list = Sample.from_lists(
            model=self.model,
            parameter_lists=parameter_list,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list
        )

        return StoredSamples(
            model=self.model,
            sample_list=sample_list,
            unconverged_sample_size=self.unconverged_sample_size,
        )
