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

    def __init__(self, initial_values=None, deltas=None, inplace=False, delta=1):
        self.initial_values = initial_values or {}
        self.inplace = inplace
        self.delta = delta
        self.deltas = deltas or {}

    def update_model_approx(
            self,
            new_model_dist: MeanField,
            factor_approx: FactorApproximation,
            model_approx: EPMeanField,
            status: Optional[Status] = Status(),
            delta: Optional[float] = None,
    ) -> Tuple[EPMeanField, Status]:
        delta = delta or self.deltas.get(factor_approx.factor) or self.delta
        new_approx, status = model_approx.project_mean_field(
            new_model_dist,
            factor_approx,
            delta=delta,
            status=status,
        )
        return new_approx, status

    @abstractmethod
    def optimise(
            self, factor: Factor, model_approx: EPMeanField, status: Status = Status()
    ) -> Tuple[EPMeanField, Status]:
        pass

    def exact_fit(
            self, factor: Factor, model_approx: EPMeanField, status: Status = Status()
    ) -> Tuple[EPMeanField, Status]:

        with LogWarnings(logger=self.logger, action='always') as caught_warnings:
            if factor._calc_exact_update:
                factor_approx = model_approx.factor_approximation(factor)
                new_approx = model_approx if self.inplace else model_approx.copy()
                new_approx.update_factor_mean_field(
                    factor, factor.calc_exact_update(factor_approx.cavity_dist)
                )
            elif factor._calc_exact_projection:
                factor_approx = model_approx.factor_approximation(factor)
                new_model_dist = factor.calc_exact_projection(factor_approx.cavity_dist)
                new_approx, status = self.update_model_approx(
                    new_model_dist, factor_approx, model_approx, status
                )

            else:
                raise NotImplementedError(
                    "Factor does not have exact updates methods"
                )

        status_kws = status._asdict()
        status_kws['messages'] = status.messages + tuple(caught_warnings.messages)
        status = Status(**status_kws)

        return new_approx, status


class ExactFactorFit(AbstractFactorOptimiser):
    optimise = AbstractFactorOptimiser.exact_fit
