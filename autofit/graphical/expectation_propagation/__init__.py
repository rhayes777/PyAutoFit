import logging
from abc import ABC, abstractmethod
from typing import (
    Dict, Tuple, Optional, List
)

import matplotlib.pyplot as plt

from autofit.graphical.factor_graphs import (
    Factor, FactorGraph
)
from autofit.graphical.utils import Status
from .ep_mean_field import EPMeanField
from .history import EPHistory, EPCallBack, FactorHistory
from ...tools.util import IntervalCounter

logger = logging.getLogger(
    __name__
)


class AbstractFactorOptimiser(ABC):
    @abstractmethod
    def optimise(
            self,
            factor: Factor,
            model_approx: EPMeanField,
            name: str = None,
            status: Status = Status()
    ) -> Tuple[EPMeanField, Status]:
        pass


class EPOptimiser:
    def __init__(
            self,
            factor_graph: FactorGraph,
            default_optimiser: Optional[AbstractFactorOptimiser] = None,
            factor_optimisers: Optional[Dict[Factor, AbstractFactorOptimiser]] = None,
            ep_history: Optional[EPHistory] = None,
            factor_order: Optional[List[Factor]] = None,
            log_interval=10,
            visualise_interval=10,
            output_interval=10
    ):
        factor_optimisers = factor_optimisers or {}
        self.factor_graph = factor_graph
        self.factors = factor_order or self.factor_graph.factors

        self.should_log = IntervalCounter(
            log_interval
        )
        self.should_visualise = IntervalCounter(
            visualise_interval
        )
        self.should_output = IntervalCounter(
            output_interval
        )

        if default_optimiser is None:
            self.factor_optimisers = factor_optimisers
            missing = set(self.factors) - self.factor_optimisers.keys()
            if missing:
                raise (ValueError(
                    f"missing optimisers for {missing}, "
                    "pass a default_optimiser or add missing optimsers"
                ))
        else:
            self.factor_optimisers = {
                factor: factor_optimisers.get(
                    factor,
                    default_optimiser
                )
                for factor in self.factors
            }

        self.ep_history = ep_history or EPHistory()

    def run(
            self,
            model_approx: EPMeanField,
            name=None,
            max_steps=100,
    ) -> EPMeanField:
        for _ in range(max_steps):

            should_log = self.should_log()
            should_visualise = self.should_visualise()
            should_output = self.should_output()

            for factor, optimiser in self.factor_optimisers.items():
                factor_logger = logging.getLogger(
                    factor.name
                )
                factor_logger.debug("Optimising...")
                try:
                    model_approx, status = optimiser.optimise(
                        factor,
                        model_approx,
                        name=name
                    )
                except (ValueError, ArithmeticError, RuntimeError) as e:
                    logger.exception(e)
                    status = Status(
                        False,
                        (f"Factor: {factor} experienced error {e}",)
                    )

                factor_logger.debug(status)

                if self.ep_history(factor, model_approx, status):
                    logger.info("Terminating optimisation")
                    break  # callback controls convergence

                if status:
                    factor_history = self.ep_history[factor]
                    log_evidence = model_approx.log_evidence
                    divergence = factor_history.kl_divergence()
                    if should_log:
                        factor_logger.info(
                            f"Log Evidence = {log_evidence}"
                        )
                        factor_logger.info(
                            f"KL Divergence = {divergence}"
                        )
                    if should_visualise:
                        plt.plot(
                            factor_history.evidences,
                            label=f"{factor.name} evidence"
                        )

            else:  # If no break do next iteration
                if should_visualise:
                    plt.legend()
                    plt.show()
                continue
            break  # stop iterations

        return model_approx
