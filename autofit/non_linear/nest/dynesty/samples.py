import dill
from dynesty.results import Results
from dynesty import utils as dyfunc
import numpy as np
from os import path
from typing import Optional

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.non_linear.samples import Sample
from autofit.non_linear.samples.nest import SamplesNest
from autofit.tools.util import open_

class SamplesDynesty(SamplesNest):

    def __add__(
            self,
            other: "SamplesDynesty"
    ) -> "SamplesDynesty":
        """
        Samples can be added together, which combines their `sample_list` meaning that inferred parameters are
        computed via their joint PDF.

        For dynesty samples, the in-built dynesty function `merge_runs` can be used to combine results_internal in their native
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

        results_internal = dyfunc.merge_runs(res_list=[self.results_internal, other.results_internal])

        return SamplesDynesty(
            model=self.model,
            sample_list=self.sample_list + other.sample_list,
            number_live_points=self._number_live_points,
            unconverged_sample_size=self.unconverged_sample_size,
            time=self.time,
            results_internal=results_internal
        )

    @classmethod
    def from_csv(cls, paths : AbstractPaths, model: AbstractPriorModel):
        """
        Returns a `Samples` object from the non-linear search output samples, which are stored in a .csv file.

        The samples object requires additional information on the non-linear search (e.g. the number of live points),
        which is extracted from the `search_info.json` file.

        This function looks for the internal results of dynesty and includes it in the samples if it exists, which
        allows for dynesty visualization tools to be used on the samples.

        Parameters
        ----------
        paths
            An object describing the paths for saving data (e.g. hard-disk directories or entries in sqlite database).
        model
            An object that represents possible instances of some model with a given dimensionality which is the number
            of free dimensions of the model.

        Returns
        -------
        The dynesty samples which have been loaded from hard-disk via .csv.
        """

        sample_list = paths.load_samples()
        samples_info = paths.load_samples_info()
        try:
            results_internal = paths.load_results_internal()
        except FileNotFoundError:
            results_internal = None

        return SamplesDynesty(
            model=model,
            sample_list=sample_list,
            number_live_points=samples_info["number_live_points"],
            unconverged_sample_size=samples_info["unconverged_sample_size"],
            time=samples_info["time"],
            results_internal=results_internal,
        )

    @classmethod
    def from_results_internal(
            cls,
            results_internal: Results,
            model: AbstractPriorModel,
            number_live_points: int,
            unconverged_sample_size: int = 100,
            time: Optional[float] = None,
    ):
        """
        Returns a `Samples` object from a Dynesty the dynesty internal results format, which contains the
        samples of the non-linear search (e.g. the parameters, log likelihoods, etc.).

        The internal dynesty results are converted from the native format used by `dynesty` to lists of values,
        for the samples.

        This classmethod performs this conversion before creating a `SamplesDynesty` object.

        Parameters
        ----------
        results_internal
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
        parameter_lists = results_internal.samples.tolist()
        log_prior_list = model.log_prior_list_from(parameter_lists=parameter_lists)
        log_likelihood_list = list(results_internal.logl)

        try:
            weight_list = list(
                np.exp(np.asarray(results_internal.logwt) - results_internal.logz[-1])
            )
        except:
            weight_list = results_internal["weights"]

        sample_list = Sample.from_lists(
            model=model,
            parameter_lists=parameter_lists,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list,
        )

        return SamplesDynesty(
            model=model,
            sample_list=sample_list,
            number_live_points=number_live_points,
            unconverged_sample_size=unconverged_sample_size,
            time=time,
            results_internal=results_internal,
        )

    @property
    def number_live_points(self):
        return self._number_live_points

    @property
    def total_samples(self):
        return int(np.sum(self.results_internal.ncall))

    @property
    def log_evidence(self):
        return np.max(self.results_internal.logz)