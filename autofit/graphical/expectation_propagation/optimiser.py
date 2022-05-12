import logging
import os
from pathlib import Path
from typing import Dict, Optional, List

from autofit import exc
from autofit.graphical.expectation_propagation.ep_mean_field import EPMeanField
from autofit.graphical.expectation_propagation.history import EPHistory
from autofit.graphical.factor_graphs.factor import Factor
from autofit.graphical.factor_graphs.graph import FactorGraph
from autofit.graphical.mean_field import Status
from autofit.graphical.utils import StatusFlag, LogWarnings
from autofit.mapper.identifier import Identifier
from autofit.non_linear.paths import DirectoryPaths
from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.tools.util import IntervalCounter
from .factor_optimiser import AbstractFactorOptimiser, ExactFactorFit
from .visualise import Visualise

logger = logging.getLogger(__name__)


class EPOptimiser:
    def __init__(
            self,
            factor_graph: FactorGraph,
            default_optimiser: Optional[AbstractFactorOptimiser] = None,
            factor_optimisers: Optional[Dict[Factor, AbstractFactorOptimiser]] = None,
            ep_history: Optional[EPHistory] = None,
            factor_order: Optional[List[Factor]] = None,
            paths: AbstractPaths = None,
            factor_loggers: Optional[Dict[Factor, logging.Logger]] = None,
    ):
        """
        Optimise a factor graph.

        Cycles through factors optimising them individually using priors
        created through expectation propagation; in effect the prior for each
        variable for a given factor is the product of the posteriors of that
        variable for all other factors.

        Parameters
        ----------
        factor_graph
            A graph describing the relationships between multiple factors
        default_optimiser
            An optimiser that is used if no specific optimiser is provided for a factor
        factor_optimisers
            Specific optimisers used to optimise each factor
        ep_history
            Optionally specify an alternate history
        factor_order
            The factors in the graph but placed in the order in which they should
            be optimised
        paths
            Optionally define how data should be output
        """
        self.paths = paths or DirectoryPaths(identifier=str(Identifier(factor_graph)))

        factor_optimisers = factor_optimisers or {}
        self.factor_graph = factor_graph
        self.factors = factor_order or self.factor_graph.factors
        self.default_optimiser = default_optimiser

        if default_optimiser is None:
            self.factor_optimisers = factor_optimisers
            missing = set(self.factors) - self.factor_optimisers.keys()
            if missing:
                raise (
                    ValueError(
                        f"missing optimisers for {missing}, "
                        "pass a default_optimiser or add missing optimsers"
                    )
                )
        else:
            self.factor_optimisers = {
                factor: factor_optimisers.get(factor, default_optimiser)
                for factor in self.factors
            }

        for optimiser in self.factor_optimisers.values():
            optimiser.paths = self.paths

        self.ep_history = ep_history or EPHistory()
        self.factor_loggers = factor_loggers or {}

        with open(self.output_path / "graph.info", "w+") as f:
            f.write(self.factor_graph.info)

        self.visualiser = Visualise(self.ep_history, self.output_path)

    @classmethod
    def from_meanfield(
            cls,
            model_approx: EPMeanField,
            default_optimiser: Optional[AbstractFactorOptimiser] = None,
            factor_optimisers: Optional[Dict[Factor, AbstractFactorOptimiser]] = None,
            ep_history: Optional[EPHistory] = None,
            factor_order: Optional[List[Factor]] = None,
            paths: AbstractPaths = None,
            factor_loggers: Optional[Dict[Factor, logging.Logger]] = None,
            exact_fit_kws=None,
    ):
        factor_graph = model_approx.factor_graph
        factor_order = factor_order or factor_graph.factors

        factor_optimisers = factor_optimisers or {}
        factor_mean_field = model_approx.factor_mean_field
        for factor in factor_graph.factors:
            if factor not in factor_optimisers:
                factor_dist = factor_mean_field[factor]
                if factor.has_exact_projection(factor_dist):
                    factor_optimisers[factor] = ExactFactorFit(**(exact_fit_kws or {}))
                else:
                    factor_optimisers[factor] = default_optimiser

        return cls(
            factor_graph=factor_graph,
            default_optimiser=default_optimiser,
            factor_optimisers=factor_optimisers,
            ep_history=ep_history,
            factor_order=factor_order,
            paths=paths,
            factor_loggers=factor_loggers,
        )

    @property
    def output_path(self) -> Path:
        """
        The path at which data will be output. Uses the name of the optimiser.

        If the path does not exist it is created.
        """
        path = Path(self.paths.output_path)
        os.makedirs(path, exist_ok=True)
        return path

    def _log_factor(self, factor: Factor):
        """
        Log information for the factor and its history.
        """
        factor_logger = logging.getLogger(factor.name)
        try:
            factor_history = self.ep_history[factor]
            if factor_history.history:
                log_evidence = factor_history.latest_update.log_evidence
                divergence = factor_history.kl_divergence()

                factor_logger.info(f"Log Evidence = {log_evidence}")
                factor_logger.info(f"KL Divergence = {divergence}")
        except exc.HistoryException as e:
            factor_logger.exception(e)

    def factor_step(
            self,
            factor,
            model_approx,
            optimiser=None
    ):
        factor_logger = self.factor_loggers.get(factor.name, logging.getLogger(factor.name))
        factor_logger.debug("Optimising...")

        optimiser = optimiser or self.factor_optimisers[factor]
        try:
            with LogWarnings(logger=factor_logger.debug, action='always') as caught_warnings:
                model_approx, status = optimiser.optimise(
                    factor,
                    model_approx,
                )

            messages = status.messages + tuple(caught_warnings.messages)

            status = Status(status.success, messages, status.flag, result=status.result)

        except (ValueError, ArithmeticError, RuntimeError) as e:
            logger.exception(e)
            status = Status(
                False,
                (f"Factor: {factor} experienced error {e}",),
                StatusFlag.FAILURE,
            )

        factor_logger.debug(status)
        return model_approx, status

    def run(
            self,
            model_approx: EPMeanField,
            max_steps: int = 100,
            log_interval: int = 10,
            visualise_interval: int = 100,
            output_interval: int = 10,
    ) -> EPMeanField:
        """
        Run the optimisation on an approximation of the model.

        Parameters
        ----------
        model_approx
            A collection of messages describing priors on the model's variables.
        max_steps
            The maximum number of steps prior to termination. Termination may also
            occur when difference in log evidence or KL Divergence drop below a given
            threshold for two consecutive optimisations of a given factor.
        log_interval
            How steps should we wait before logging information?
        visualise_interval
            How steps should we wait before visualising information?
            This includes plots of KL Divergence and Evidence.
        output_interval
            How steps should we wait before outputting information?
            This includes the model.results file which describes the current mean values
            of each message.

        Returns
        -------
        An updated approximation of the model
        """
        should_log = IntervalCounter(log_interval)
        should_visualise = IntervalCounter(visualise_interval)
        should_output = IntervalCounter(output_interval)

        for _ in range(max_steps):
            _should_log = should_log()
            _should_visualise = should_visualise()
            _should_output = should_output()
            for factor, optimiser in self.factor_optimisers.items():
                model_approx, status = self.factor_step(
                    factor, model_approx, optimiser
                )
                if status and _should_log:
                    self._log_factor(factor)

                if self.ep_history(factor, model_approx, status):
                    logger.info("Terminating optimisation")
                    break  # callback controls convergence

            else:  # If no break do next iteration
                if _should_visualise:
                    self.visualiser()
                if _should_output:
                    self._output_results(model_approx)
                continue
            break  # stop iterations

        self.visualiser()
        self._output_results(model_approx)

        return model_approx

    def _output_results(self, model_approx: EPMeanField):
        """
        Save the graph.results text
        """
        with open(self.output_path / "graph.results", "w+") as f:
            f.write(self.factor_graph.make_results_text(model_approx))
