import logging
from functools import wraps
from typing import (
    Tuple, Callable, List, Union
)

import numpy as np

from autofit.graphical.factor_graphs import (
    Factor
)
from autofit.graphical.utils import Status
from .ep_mean_field import EPMeanField

logger = logging.getLogger(
    __name__
)

EPCallBack = Callable[[Factor, EPMeanField, Status], bool]


def default_inf(func):
    """
    Decorator that catches IndexError and returns inf.

    This used to give infinite divergence when there is insufficient
    history for a given factor.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except IndexError:
            return float("inf")

    return wrapper


class FactorHistory:
    def __init__(
            self,
            factor: Factor
    ):
        """
        Tracks the history of a single factor

        Parameters
        ----------
        factor
            A factor in a graph undergoing optimisation
        """
        self.factor = factor
        self.history = list()

    def __call__(
            self,
            approx: EPMeanField,
            status: Status
    ):
        """
        Add an optimisation result to the factor's history

        Parameters
        ----------
        approx
            The posterior mean field for an optimisation of the factor
        status
            Describes whether the optimisation was successful
        """
        self.history.append((
            approx, status
        ))

    @property
    def successes(self) -> List[EPMeanField]:
        """
        A list of mean fields produced by successful optimisations
        """
        return [
            approx for approx, success
            in self.history if success
        ]

    @property
    def latest_successful(self) -> EPMeanField:
        """
        A mean field for the last successful optimisation
        """
        return self.successes[-1]

    @property
    def previous_successful(self) -> EPMeanField:
        """
        A mean field for the last-but-one successful optimisation
        """
        return self.successes[-2]

    @default_inf
    def kl_divergence(self) -> Union[float, np.ndarray]:
        """
        The KL Divergence between the mean fields produced by the last
        two successful optimisations.

        If there are less than two successful optimisations then this is
        infinite.
        """
        return self.latest_successful.mean_field.kl(
            self.previous_successful.mean_field
        )

    @default_inf
    def evidence_divergence(self) -> Union[float, np.ndarray]:
        """
        The difference in the evidences between produced by the last two
        successful optimisations.

        If there are less than two successful optimisations then this is
        infinite.
        """
        return self.latest_successful.log_evidence - self.previous_successful.log_evidence


class EPHistory:
    def __init__(
            self,
            callbacks: Tuple[EPCallBack, ...] = (),
            kl_tol=1e-1,
            evidence_tol=None
    ):
        self._callbacks = callbacks
        self.history = {}

        self.kl_tol = kl_tol
        self.evidence_tol = evidence_tol

    def __getitem__(self, factor: Factor):
        try:
            return self.history[factor]
        except KeyError:
            self.history[factor] = FactorHistory(factor)
            return self.history[factor]

    def __call__(
            self,
            factor: Factor,
            approx: EPMeanField,
            status: Status = Status()
    ) -> bool:
        self[factor](approx, status)
        if status.success:
            if any([
                callback(factor, approx, status)
                for callback in self._callbacks
            ]):
                return True

            return self.is_converged(factor)

        return False

    def is_kl_converged(
            self,
            factor: Factor
    ) -> bool:
        return self[factor].kl_divergence() < self.kl_tol

    def is_kl_evidence_converged(
            self,
            factor: Factor
    ) -> bool:
        evidence_divergence = self[factor].evidence_divergence()

        if evidence_divergence < 0:
            logger.warning(
                f"Evidence for factor {factor} has decreased"
            )

        return abs(evidence_divergence) < self.evidence_tol

    def is_converged(
            self,
            factor: Factor
    ) -> bool:
        if self.kl_tol and self.is_kl_converged(factor):
            return True
        if self.evidence_tol and self.is_kl_evidence_converged(factor):
            return True
        return False
