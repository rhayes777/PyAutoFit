import numpy as np
from typing import List, Optional

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.samples import Samples, Sample


class PySwarmsSamples(Samples):

    def __init__(
            self,
            model: AbstractPriorModel,
            sample_list: List[Sample],
            total_iterations: int,
            time: Optional[float] = None,
            results_internal: Optional = None,
    ):
        """
        The `Samples` classes in **PyAutoFit** provide an interface between the results_internal of
        a `NonLinearSearch` (e.g. as files on your hard-disk) and Python.

        For example, the output class can be used to load an instance of the best-fit model, get an instance of any
        individual sample by the `NonLinearSearch` and return information on the likelihoods, errors, etc.

        This class stores the samples of optimizer model-fits (e.g. `pyswarms`, `LBFGS`). To use a library's
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
        total_iterations
            The total number of PySwarms iterations, which cannot be estimated from the sample list (which contains
            only accepted samples).
        time
            The time taken to perform the model-fit, which is passed around `Samples` objects for outputting
            information on the overall fit.
        results_internal
            The nested sampler's results in their native internal format for interfacing its visualization library.
        """

        self.results_internal = results_internal
        self.total_iterations = total_iterations

        super().__init__(
            model=model,
            sample_list=sample_list,
            time=time,
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

        This classmethod performs this conversion before creating a `PySwarmsSamples` object.

        Parameters
        ----------
        results_internal
            The MCMC results in their native internal format from which the samples are computed.
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

        return PySwarmsSamples(
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