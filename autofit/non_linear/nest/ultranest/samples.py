from typing import List, Optional
import warnings

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.samples import Sample
from autofit.non_linear.samples.nest import NestSamples

from autofit import exc

class UltraNestSamples(NestSamples):

    def __init__(
            self,
            model: AbstractPriorModel,
            sample_list: List[Sample],
            number_live_points: int,
            unconverged_sample_size: int = 100,
            time: Optional[float] = None,
            results: Optional = None,
    ):
        """
        The `Samples` classes in **PyAutoFit** provide an interface between the results of a `NonLinearSearch` (e.g.
        as files on your hard-disk) and Python.

        For example, the output class can be used to load an instance of the best-fit model, get an instance of any
        individual sample by the `NonLinearSearch` and return information on the likelihoods, errors, etc.

        This class stores the samples of a model-fit using `UltraNest`. In order to use the in-built `UltraNest`
        visualization tools the results are optionally stored in their native format internal to `UltraNest` using
        the `results` attribute.

        Parameters
        ----------
        model
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        sample_list
            The list of `Samples` which contains the paramoeters, likelihood, weights, etc. of every sample taken
            by the non-linear search.
        number_live_points
            The number of live points used by the `UltraNest` search.
        unconverged_sample_size
            If the samples are for a search that is yet to convergence, a reduced set of samples are used to provide
            a rough estimate of the parameters. The number of samples is set by this parameter.
        time
            The time taken to perform the model-fit, which is passed around `Samples` objects for outputting
            information on the overall fit.
        results
            The `UltraNest` results in their native internal format for interfacing the UltraNest visualization library.
        """

        self.results = results
        self._number_live_points = number_live_points

        super().__init__(
            model=model,
            sample_list=sample_list,
            unconverged_sample_size=unconverged_sample_size,
            time=time,
        )

    def __add__(
            self,
            other: "UltraNestSamples"
    ) -> "UltraNestSamples":
        """
        Samples can be added together, which combines their `sample_list` meaning that inferred parameters are
        computed via their joint PDF.

        For UltraNest samples there are no tools for combining results in their native format, therefore these results
        are set to None and support for visualization is disabled.

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
            "Addition of UltraNestSamples cannot retain results in native format. "
            "Visualization of summed samples diabled.",
            exc.SamplesWarning
        )

        return UltraNestSamples(
            model=self.model,
            sample_list=self.sample_list + other.sample_list,
            number_live_points=self._number_live_points,
            unconverged_sample_size=self.unconverged_sample_size,
            time=self.time,
            results=None
        )

    @classmethod
    def from_results(
            cls,
            results,
            model: AbstractPriorModel,
            number_live_points: int,
            unconverged_sample_size: int = 100,
            time: Optional[float] = None,
    ):
        """
        The `Samples` classes in **PyAutoFit** provide an interface between the results of a `NonLinearSearch` (e.g.
        as files on your hard-disk) and Python.

        To create a `Samples` object after an `UltraNest` model-fit the results must be converted from the
        native format used by `UltraNest` to lists of values, the format used by the **PyAutoFit** `Samples` objects.
        This classmethod performs this conversion before creating a DyenstySamples` object.

        Parameters
        ----------
        results
            The `UltraNest` results in their native internal format from which the samples are computed.
        model
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        number_live_points
            The number of live points used by the `UltraNest` search.
        unconverged_sample_size
            If the samples are for a search that is yet to convergence, a reduced set of samples are used to provide
            a rough estimate of the parameters. The number of samples is set by this parameter.
        time
            The time taken to perform the model-fit, which is passed around `Samples` objects for outputting
            information on the overall fit.
        """
        
        parameters = results["weighted_samples"]["points"]
        log_likelihood_list = results["weighted_samples"]["logl"]
        log_prior_list = [
            sum(model.log_prior_list_from_vector(vector=vector)) for vector in parameters
        ]
        weight_list = results["weighted_samples"]["weights"]

        sample_list = Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list
        )

        return UltraNestSamples(
            model=model,
            sample_list=sample_list,
            number_live_points=number_live_points,
            unconverged_sample_size=unconverged_sample_size,
            time=time,
            results=results,
        )

    @property
    def number_live_points(self):
        return self._number_live_points

    @property
    def total_samples(self):
        return self.results["ncall"]

    @property
    def log_evidence(self):
        return self.results["logz"]
