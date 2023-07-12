import json
import numpy as np
from os import path
import pickle
from typing import Optional

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.non_linear.samples import Samples, Sample
from autofit.tools.util import open_

class SamplesPySwarms(Samples):

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

        with open_(path.join(paths.search_internal_path, "results_internal.json"), "r+") as f:
            results_internal_dict = json.load(f)

        try:
            with open_(path.join(paths.search_internal_path, "results_internal.pickle"), "rb") as f:
                results_internal = pickle.load(f)
        except FileNotFoundError:
            results_internal = None

        return SamplesPySwarms(
            model=model,
            sample_list=sample_list,
            total_iterations=results_internal_dict["total_iterations"],
            time=samples_info["time"],
            results_internal=results_internal,
        )

    @classmethod
    def from_results_internal(
            cls,
            results_internal: np.ndarray,
            log_posterior_list: np.ndarray,
            model: AbstractPriorModel,
            total_iterations: int,
            time: Optional[float] = None,
    ):
        """
        The `Samples` classes in **PyAutoFit** provide an interface between the results of a `NonLinearSearch` (e.g.
        as files on your hard-disk) and Python.

        To create a `Samples` object after an `pyswarms` model-fit the results must be converted from the
        native format used by `pyswarms` (which are numpy ndarrays) to lists of values, the format used by
        the **PyAutoFit** `Samples` objects.

        This classmethod performs this conversion before creating a `SamplesPySwarms` object.

        Parameters
        ----------
        results_internal
            The Pyswarms results in their native internal format from which the samples are computed.
        log_posterior_list
            The log posterior of the PySwarms accepted samples.
        model
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        total_iterations
            The total number of PySwarms iterations, which cannot be estimated from the sample list (which contains
            only accepted samples).
        time
            The time taken to perform the model-fit, which is passed around `Samples` objects for outputting
            information on the overall fit.
        """
        parameter_lists = [
            param.tolist() for parameters in results_internal for param in parameters
        ]
        log_prior_list = model.log_prior_list_from(parameter_lists=parameter_lists)
        log_likelihood_list = [lp - prior for lp, prior in zip(log_posterior_list, log_prior_list)]
        weight_list = len(log_likelihood_list) * [1.0]

        sample_list = Sample.from_lists(
            model=model,
            parameter_lists=[parameters.tolist()[0] for parameters in results_internal],
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list
        )

        return SamplesPySwarms(
            model=model,
            sample_list=sample_list,
            total_iterations=total_iterations,
            time=time,
            results_internal=results_internal,
        )

    @property
    def points(self):
        """
        Makes internal results accessible as `self.points` for consistency with PySwarms API.
        """
        return self.results_internal