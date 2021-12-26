import numpy as np
from typing import Optional

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.samples import Samples, Sample


class PySwarmsSamples(Samples):

    def __init__(
            self,
            model: AbstractPriorModel,
            points: np.ndarray,
            log_posterior_list: np.ndarray,
            total_iterations: int,
            time: Optional[float] = None,
    ):
        """
        Create an *Samples* object from this non-linear search's output files on the hard-disk and model.

        For PySwarms, all quantities are extracted via pickled states of the particle and cost histories.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space. This maps the points from unit
            cube values to physical values via the priors.
        """

        self.points = points
        self._log_posterior_list = log_posterior_list
        self.total_iterations = total_iterations

        parameter_lists = [
            param.tolist() for parameters in self.points for param in parameters
        ]
        log_prior_list = model.log_prior_list_from(parameter_lists=parameter_lists)
        log_likelihood_list = [lp - prior for lp, prior in zip(self._log_posterior_list, log_prior_list)]
        weight_list = len(log_likelihood_list) * [1.0]

        sample_list = Sample.from_lists(
            model=model,
            parameter_lists=[parameters.tolist()[0] for parameters in self.points],
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list
        )

        super().__init__(
            model=model,
            sample_list=sample_list,
            time=time,
        )
