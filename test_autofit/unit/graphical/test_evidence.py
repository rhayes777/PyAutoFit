import numpy as np
import pytest
from scipy import stats, integrate

import autofit.graphical.messages.normal
from autofit import graphical as graph



def test_normal_log_norm():
    m1 = graph.NormalMessage(0.5, 1.3)
    m2 = graph.NormalMessage(-0.1, 0.2)

    A = np.exp(m1.log_normalisation(m2))
    i12, ierr = integrate.quad(
        lambda x: m1.pdf(x) * m2.pdf(x), -np.inf, np.inf)

    assert np.abs(A - i12) < ierr < 1e-6


def test_gamma_log_norm():
    m1 = graph.GammaMessage(1.5, 1.3)
    m2 = graph.GammaMessage(1.3, 1.2)

    A = np.exp(m1.log_normalisation(m2))
    i12, ierr = integrate.quad(
        lambda x: m1.pdf(x) * m2.pdf(x), 0, np.inf)

    assert np.abs(A - i12) < ierr < 1e-6