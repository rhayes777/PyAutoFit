import numpy as np
import pytest
from scipy import stats, integrate

import autofit.graphical.messages.normal
from autofit import graphical as graph

def test_message_norm():
    messages = [
        tuple(
            map(graph.NormalMessage, 
                [0.5, 0.1], [0.2, 0.3])),
        tuple(
            map(graph.NormalMessage, 
                [0.5, 0.1], [0.2, 0.3], [-0.5, 1.3])),
        tuple(
            map(graph.GammaMessage, 
                [0.5, 1.1], [0.2, 1.3])),
        tuple(
            map(graph.GammaMessage, 
                [0.5, 1.1], [0.2, 1.3], [2, 1])),
    ]
    for ms in messages:
        m1, *m2s = ms
        A = np.exp(m1.log_normalisation(*m2s))

        # Calculate normalisation numerically
        i12, ierr = integrate.quad(
            lambda x: np.exp(sum(m.logpdf(x) for m in ms)), 
            *m1._support[0])

        # verify within tolerance
        print(ms)
        assert np.abs(A - i12) < ierr < 1e-6