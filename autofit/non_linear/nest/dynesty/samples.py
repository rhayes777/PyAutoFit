from typing import Optional, List

import numpy as np
from dynesty.results import Results
from dynesty import utils as dyfunc

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.samples import Sample
from autofit.non_linear.samples.nest import NestSamples


class DynestySamples(NestSamples):

    def __init__(
            self,
            model: AbstractPriorModel,
            sample_list: List[Sample],
            number_live_points: int,
            unconverged_sample_size: int = 100,
            time: Optional[float] = None,
            results: Optional[Results] = None,
    ):
        """
        The `Samples` classes in **PyAutoFit** provide an interface between the results of a `NonLinearSearch` (e.g.
        as files on your hard-disk) and Python.

        For example, the output class can be used to load an instance of the best-fit model, get an instance of any
        individual sample by the `NonLinearSearch` and return information on the likelihoods, errors, etc.

        This class stores the samples of a model-fit using `Dynesty`. In order to use the in-built `dynesty`
        visualization tools the results are optionally stored in their native format internal to `dynesty` using
        the `results` attribute.

        Parameters
        ----------
        model
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        sample_list
            The list of `Samples` which contains the paramoeters, likelihood, weights, etc. of every sample taken
            by the non-linear search.
        number_live_points
            The number of live points used by the `dynesty` search.
        unconverged_sample_size
            If the samples are for a search that is yet to convergence, a reduced set of samples are used to provide
            a rough estimate of the parameters. The number of samples is set by this parameter.
        time
            The time taken to perform the model-fit, which is passed around `Samples` objects for outputting
            information on the overall fit.
        results
            The `dynesty` results in their native internal format for interfacing the Dynesty visualization library.
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
            other: "DynestySamples"
    ) -> "DynestySamples":
        """
        Samples can be added together, which combines their `sample_list` meaning that inferred parameters are
        computed via their joint PDF.

        For dynesty samples, the in-built dynesty function `merge_runs` can be used to combine results in their native
        format and therefore retain visualization support.

        Parameters
        ----------
        other
            Another Samples class

        Returns
        -------
        A class that combined the samples of the two Samples objects.
        """

        self._check_addition(other=other)

        results = dyfunc.merge_runs(res_list=[self.results, other.results])

        return DynestySamples(
            model=self.model,
            sample_list=self.sample_list + other.sample_list,
            number_live_points=self._number_live_points,
            unconverged_sample_size=self.unconverged_sample_size,
            time=self.time,
            results=results
        )

    @classmethod
    def from_results(
            cls,
            results: Results,
            model: AbstractPriorModel,
            number_live_points: int,
            unconverged_sample_size: int = 100,
            time: Optional[float] = None,
    ):
        """
        The `Samples` classes in **PyAutoFit** provide an interface between the results of a `NonLinearSearch` (e.g.
        as files on your hard-disk) and Python.

        To create a `Samples` object after a `dynesty` model-fit the results must be converted from the
        native format used by `dynesty` to lists of values, the format used by the **PyAutoFit** `Samples` objects.
        This classmethod performs this conversion before creating a DyenstySamples` object.

        Parameters
        ----------
        results
            The `dynesty` results in their native internal format from which the samples are computed.
        model
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        number_live_points
            The number of live points used by the `dynesty` search.
        unconverged_sample_size
            If the samples are for a search that is yet to convergence, a reduced set of samples are used to provide
            a rough estimate of the parameters. The number of samples is set by this parameter.
        time
            The time taken to perform the model-fit, which is passed around `Samples` objects for outputting
            information on the overall fit.
        """
        parameter_lists = results.samples.tolist()
        log_prior_list = model.log_prior_list_from(parameter_lists=parameter_lists)
        log_likelihood_list = list(results.logl)

        try:
            weight_list = list(
                np.exp(np.asarray(results.logwt) - results.logz[-1])
            )
        except:
            weight_list = results["weights"]

        sample_list = Sample.from_lists(
            model=model,
            parameter_lists=parameter_lists,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list,
        )

        return DynestySamples(
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
        return int(np.sum(self.results.ncall))

    @property
    def log_evidence(self):
        return np.max(self.results.logz)
