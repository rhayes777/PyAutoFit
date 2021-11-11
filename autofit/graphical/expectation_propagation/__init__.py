import logging
from abc import ABC, abstractmethod
from typing import (
    Dict, Tuple, Optional, List
)

from autofit.graphical.factor_graphs import (
    Factor, FactorGraph
)
from autofit.graphical.utils import Status
from .ep_mean_field import EPMeanField
from .history import EPHistory, EPCallBack, FactorHistory

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
            callback: Optional[EPCallBack] = None,
            factor_order: Optional[List[Factor]] = None
    ):
        factor_optimisers = factor_optimisers or {}
        self.factor_graph = factor_graph
        self.factors = factor_order or self.factor_graph.factors

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

        self.callback = callback or EPHistory()

    def run(
            self,
            model_approx: EPMeanField,
            name=None,
            max_steps=100,
    ) -> EPMeanField:
        for _ in range(max_steps):
            for factor, optimiser in self.factor_optimisers.items():
                try:
                    model_approx, status = optimiser.optimise(
                        factor,
                        model_approx,
                        name=name
                    )
                except TypeError as e:
                    raise e
                except (ValueError, ArithmeticError, RuntimeError) as e:
                    status = Status(
                        False,
                        (f"Factor: {factor} experienced error {e}",)
                    )

                if self.callback(factor, model_approx, status):
                    break  # callback controls convergence
            else:  # If no break do next iteration
                continue
            break  # stop iterations

        return model_approx
