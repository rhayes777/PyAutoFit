import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Generator

import matplotlib.pyplot as plt

from autofit import exc
from autofit.graphical.expectation_propagation.ep_mean_field import EPMeanField
from autofit.graphical.expectation_propagation.history import EPHistory
from autofit.graphical.factor_graphs.factor import Factor
from autofit.graphical.factor_graphs.graph import FactorGraph
from autofit.graphical.mean_field import MeanField, FactorApproximation, Status
from autofit.graphical.utils import StatusFlag, LogWarnings
from autofit.mapper.identifier import Identifier
from autofit.mapper.variable import Plate
from autofit.non_linear.paths import DirectoryPaths
from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.tools.util import IntervalCounter

logger = logging.getLogger(__name__)


class AbstractFactorOptimiser(ABC):
    """
    An optimiser used to optimise individual factors during EPOptimisation.
    """
    logger = logger.debug

    def __init__(self, initial_values=None, deltas=None, inplace=False):
        self.initial_values = initial_values or {}
        self.inplace = inplace
        self.deltas = defaultdict(lambda: 1)
        if deltas:
            self.deltas.update(deltas)

    def update_model_approx(
            self,
            new_model_dist: MeanField,
            factor_approx: FactorApproximation,
            model_approx: EPMeanField,
            status: Optional[Status] = Status(),
            delta: Optional[float] = None,
    ) -> Tuple[EPMeanField, Status]:
        delta = delta or self.deltas[factor_approx.factor]
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


class Visualise:
    def __init__(self, ep_history: EPHistory, output_path: Path):
        """
        Handles visualisation of expectation propagation optimisation.

        This includes plotting key metrics such as Evidence and KL Divergence
        which are expected to converge.

        Parameters
        ----------
        ep_history
            A history describing previous optimisations by factor
        output_path
            The path that plots are written to
        """
        self.ep_history = ep_history
        self.output_path = output_path

    def __call__(self):
        """
        Save a plot of Evidence and KL Divergence for the ep_history
        """
        fig, (evidence_plot, kl_plot) = plt.subplots(2)
        fig.suptitle("Evidence and KL Divergence")
        evidence_plot.plot(self.ep_history.evidences(), label="evidence")
        kl_plot.semilogy(self.ep_history.kl_divergences(), label="KL divergence")
        # for factor, factor_history in self.ep_history.items():
        #     evidence_plot.plot(
        #         factor_history.evidences, label=f"{factor.name} evidence"
        #     )
        #     kl_plot.plot(
        #         factor_history.kl_divergences, label=f"{factor.name} divergence"
        #     )
        evidence_plot.legend()
        kl_plot.legend()
        plt.savefig(str(self.output_path / "graph.png"))


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

            status = Status(status.success, messages, status.flag)

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
            for factor, optimiser in self.factor_optimisers.items():
                model_approx, status = self.factor_step(
                    factor, model_approx, optimiser
                )
                if status and should_log():
                    self._log_factor(factor)

                if self.ep_history(factor, model_approx, status):
                    logger.info("Terminating optimisation")
                    break  # callback controls convergence

            else:  # If no break do next iteration
                if should_visualise():
                    self.visualiser()
                if should_output():
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


class StochasticEPOptimiser(EPOptimiser):

    def factor_step(
            self,
            factor,
            subset_approx,
            optimiser=None
    ):
        factor_logger = self.factor_loggers.get(factor.name, logging.getLogger(factor.name))
        factor_logger.debug("Optimising...")

        optimiser = optimiser or self.factor_optimisers[factor]
        subset_factor = subset_approx._factor_subset_factor[factor]
        try:
            with LogWarnings(logger=factor_logger.debug, action='always') as caught_warnings:

                subset_approx, status = optimiser.optimise(
                    subset_factor,
                    subset_approx,
                )

            messages = status.messages + tuple(caught_warnings.messages)
            status = Status(status.success, messages, status.flag)
        except (ValueError, ArithmeticError, RuntimeError) as e:
            logger.exception(e)
            status = Status(
                False,
                status.messages + (f"Factor: {factor} experienced error {e}",),
                StatusFlag.FAILURE,
            )

        factor_logger.debug(status)
        return subset_approx, status

    def run(
            self,
            model_approx: EPMeanField,
            batches: Generator[Dict[Plate, List[int]], None, None],
            log_interval: int = 10,
            visualise_interval: int = 100,
            output_interval: int = 10,
            inplace=False,
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

        for batch in batches:
            subset_approx = model_approx[batch]

            for factor, optimiser in self.factor_optimisers.items():
                subset_approx, status = self.factor_step(
                    factor, subset_approx, optimiser
                )

                if status and should_log():
                    self._log_factor(factor)

                if self.ep_history(factor, subset_approx, status):
                    logger.info("Terminating optimisation")
                    break  # callback controls convergence

            else:  # If no break do next iteration

                if inplace:
                    model_approx[batch] = subset_approx
                else:
                    model_approx = model_approx.merge(batch, subset_approx)

                if self.ep_history(
                        model_approx.factor_graph, model_approx
                ):
                    logger.info("Terminating optimisation")
                    break  # callback controls convergence

                if should_visualise():
                    self.visualiser()
                if should_output():
                    self._output_results(model_approx)
                continue
            break  # stop iterations

        self.visualiser()
        self._output_results(model_approx)

        return model_approx
