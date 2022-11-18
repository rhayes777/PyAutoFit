import logging
from abc import ABC, abstractmethod
from typing import Tuple, Optional

from autofit.graphical.expectation_propagation.ep_mean_field import EPMeanField
from autofit.graphical.factor_graphs.factor import Factor
from autofit.graphical.mean_field import MeanField, FactorApproximation, Status
from autofit.graphical.utils import LogWarnings

logger = logging.getLogger(__name__)


class AbstractFactorOptimiser(ABC):
    """
    An optimiser used to optimise individual factors during EPOptimisation.
    """
    logger = logger.debug

    def __init__(self, initial_values=None, deltas=None, inplace=False, delta=1, dynamic_delta=False):
        self.initial_values = initial_values or {}
        self.inplace = inplace
        self.delta = delta
        self.deltas = deltas or {}
        self.dynamic_delta = dynamic_delta

    def update_model_approx(
            self,
            new_model_dist: MeanField,
            factor_approx: FactorApproximation,
            model_approx: EPMeanField,
            status: Optional[Status] = Status(),
            delta: Optional[float] = None,
    ) -> Tuple[EPMeanField, Status]:

        variable_message_count = model_approx.variable_message_count
        min_value = min(variable_message_count.values())

        delta = delta or self.delta

        if factor_approx.factor in self.deltas:
            delta = self.deltas[factor_approx.factor]
        elif self.dynamic_delta:
            delta = MeanField({
                variable: self.delta * (min_value / message_count)
                for variable, message_count in variable_message_count.items()
            })

        return model_approx.project_mean_field(
            new_model_dist,
            factor_approx,
            delta=delta,
            status=status,
        )

    @abstractmethod
    def optimise(
            self, factor_approx: FactorApproximation, status: Status = Status()
    ) -> Tuple[MeanField, Status]:
        pass

    def exact_fit(
            self, factor_approx: FactorApproximation, status: Status = Status()
    ) -> Tuple[MeanField, Status]:
        factor = factor_approx.factor
        cavity_dist = factor_approx.cavity_dist
        with LogWarnings(logger=self.logger, action='always') as caught_warnings:
            if factor._calc_exact_update:
                factor_mean_field = factor.calc_exact_update(cavity_dist)
                new_model_dist = factor_mean_field * cavity_dist
            elif factor._calc_exact_projection:
                new_model_dist = factor.calc_exact_projection(cavity_dist)
            else:
                raise NotImplementedError(
                    "Factor does not have exact updates methods"
                )

        return new_model_dist, status


class ExactFactorFit(AbstractFactorOptimiser):
    optimise = AbstractFactorOptimiser.exact_fit
