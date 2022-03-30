import math
import warnings
from typing import Union

import numpy as np
from scipy.special import betaln
from scipy.special import psi, polygamma

from autoconf import cached_property
from ..messages.abstract import AbstractMessage


def grad_betaln(ab):
    psiab = psi(ab.sum(axis=1, keepdims=True))
    return psi(ab) - psiab


def jac_grad_betaln(ab):
    psi1ab = polygamma(1, ab.sum(axis=1, keepdims=True))
    fii = polygamma(1, ab) - psi1ab
    fij = -psi1ab[:, 0]
    return np.array([[fii[:, 0], fij], [fij, fii[:, 1]]]).T


def inv_beta_suffstats(lnX, ln1X):
    """Solve for a, b for,

    psi(a) + psi(a + b) = lnX
    psi(b) + psi(a + b) = ln1X
    """
    _lnX, _ln1X = np.ravel(lnX), np.ravel(ln1X)
    lnXs = np.c_[_lnX, _ln1X]

    # Find initial starting location
    Gs = np.exp(lnXs)
    dG = 1 - Gs.sum(axis=1, keepdims=True)
    ab = np.maximum(1, (1 + Gs / dG) / 2)

    # 5 Newton Raphson itertions is generally enough
    for i in range(5):
        f = grad_betaln(ab) - lnXs
        jac = jac_grad_betaln(ab)
        ab += np.linalg.solve(jac, - f)

    if np.any(ab < 0):
        warnings.warn(
            "invalid negative parameters found for inv_beta_suffstats, "
            "clampling value to 0.5",
            RuntimeWarning
        )
        b = np.clip(ab, 0.5, None)

    shape = np.shape(lnX)
    if shape:
        a = ab[:, 0].reshape(shape)
        b = ab[:, 1].reshape(shape)
    else:
        a, b = ab[0, :]

    return a, b


class BetaMessage(AbstractMessage):
    """
    Models a Beta distribution
    """
    log_base_measure = 0
    _support = ((0, 1),)
    _min = 0
    _max = 1
    _range = 1
    _parameter_support = ((0, np.inf), (0, np.inf))

    def __init__(
            self,
            alpha=0.5,
            beta=0.5,
            lower_limit=-math.inf,
            upper_limit=math.inf,
            log_norm=0,
            id_=None
    ):
        self.alpha = alpha
        self.beta = beta
        super().__init__(
            alpha,
            beta,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            log_norm=log_norm,
            id_=id_
        )

    def value_for(self, unit: float) -> float:
        raise NotImplemented()

    @cached_property
    def log_partition(self) -> np.ndarray:
        return betaln(*self.parameters)

    @cached_property
    def natural_parameters(self) -> np.ndarray:
        return self.calc_natural_parameters(
            self.alpha,
            self.beta
        )

    @staticmethod
    def calc_natural_parameters(
            alpha: Union[float, np.ndarray],
            beta: Union[float, np.ndarray]
    ) -> np.ndarray:
        return np.array([alpha - 1, beta - 1])

    @staticmethod
    def invert_natural_parameters(
            natural_parameters: np.ndarray
    ) -> np.ndarray:
        return natural_parameters + 1

    @classmethod
    def invert_sufficient_statistics(
            cls, sufficient_statistics: np.ndarray
    ) -> np.ndarray:
        a, b = inv_beta_suffstats(*sufficient_statistics)
        return cls.calc_natural_parameters(a, b)

    @classmethod
    def to_canonical_form(cls, x: np.ndarray) -> np.ndarray:
        return np.array([np.log(x), np.log1p(-x)])

    @cached_property
    def mean(self) -> Union[np.ndarray, float]:
        return self.alpha / (self.alpha + self.beta)

    @cached_property
    def variance(self) -> Union[np.ndarray, float]:
        return (
                self.alpha * self.beta
                / (self.alpha + self.beta) ** 2
                / (self.alpha + self.beta + 1)
        )

    def sample(self, n_samples=None):
        a, b = self.parameters
        shape = (n_samples,) + self.shape if n_samples else self.shape
        return np.random.beta(a, b, size=shape)

    def kl(self, dist):
        # TODO check this is correct
        # https://arxiv.org/pdf/0911.4863.pdf
        if self._support != dist._support:
            raise TypeError('Support does not match')

        aP, bP = dist.parameters
        aQ, bQ = self.parameters
        return (
                betaln(aQ, bQ) - betaln(aP, bP)
                - (aQ - aP) * psi(aP)
                - (bQ - bP) * psi(bP)
                + (aQ - aP + bQ - bP) * psi(aP + bP)
        )

    def logpdf_gradient(self, x):
        logl = self.logpdf(x)
        a, b = self.parameters
        gradl = (a - 1) / x + (b - 1) / (x - 1)
        return logl, gradl

    def logpdf_gradient_hessian(self, x):
        logl = self.logpdf(x)
        a, b = self.parameters
        ax, bx = (a - 1) / x, (b - 1) / (x - 1)
        gradl = ax + bx
        hessl = -ax / x - bx / (x - 1)
        return logl, gradl, hessl
