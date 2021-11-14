from os import path
from typing import List, Optional

import numpy as np
from sqlalchemy.orm import Session

from autoconf import conf
from autofit import exc
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.optimize.abstract_optimize import AbstractOptimizer
from autofit.non_linear.samples import OptimizerSamples, Sample
from autofit.non_linear.abstract_search import PriorPasser
from autofit.non_linear.initializer import Initializer
from autofit.plot import DrawerPlotter
from autofit.plot.mat_wrap.wrap.wrap_base import Output


class DrawerSamples(OptimizerSamples):

    def __init__(
            self,
            model: AbstractPriorModel,
            parameter_lists: List[List[float]],
            log_posterior_list: List[float],
            time: Optional[float] = None,
    ):
        """
        Create an *OptimizerSamples* object from this non-linear search's output files on the hard-disk and model.

        For Drawer, all quantities are extracted via pickled states of the particle and cost histories.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space. This maps the points from unit
            cube values to physical values via the priors.
        """

        self._log_posterior_list = log_posterior_list

        log_prior_list = [
            sum(model.log_prior_list_from_vector(vector=vector)) for vector in parameter_lists
        ]
        log_likelihood_list = [lp - prior for lp, prior in zip(self._log_posterior_list, log_prior_list)]
        weight_list = len(log_likelihood_list) * [1.0]

        sample_list = Sample.from_lists(
            model=model,
            parameter_lists=parameter_lists,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list
        )

        super().__init__(
            model=model,
            sample_list=sample_list,
            time=time,
        )
