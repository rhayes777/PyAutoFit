import pickle
from os import path
from typing import Optional

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.non_linear.samples import Sample
from autofit.non_linear.samples.nest import SamplesNest
from autofit.tools.util import open_


class SamplesUltraNest(SamplesNest):

    @classmethod
    def from_csv(cls, paths: AbstractPaths, model: AbstractPriorModel):
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
            with open_(path.join(paths.search_internal_path, "results_internal.pickle"), "rb") as f:
                results_internal = pickle.load(f)
        except FileNotFoundError:
            results_internal = None

        return SamplesUltraNest(
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
            results_internal,
            model: AbstractPriorModel,
            number_live_points: int,
            unconverged_sample_size: int = 100,
            time: Optional[float] = None,
    ):
        """
        The `Samples` classes in **PyAutoFit** provide an interface between the resultsof a `NonLinearSearch` (e.g.
        as files on your hard-disk) and Python.

        To create a `Samples` object after an `UltraNest` model-fit the results must be converted from the
        native format used by `UltraNest` to lists of values, the format used by the **PyAutoFit** `Samples` objects.
        This classmethod performs this conversion before creating a DyenstySamples` object.

        Parameters
        ----------
        results_internal
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

        parameters = results_internal["weighted_samples"]["points"]
        log_likelihood_list = results_internal["weighted_samples"]["logl"]
        log_prior_list = [
            sum(model.log_prior_list_from_vector(vector=vector)) for vector in parameters
        ]
        weight_list = results_internal["weighted_samples"]["weights"]

        sample_list = Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list
        )

        return SamplesUltraNest(
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
        return self.results_internal["ncall"]

    @property
    def log_evidence(self):
        return self.results_internal["logz"]