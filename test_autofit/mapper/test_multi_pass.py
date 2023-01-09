import os
from pathlib import Path
from random import random

import autofit as af
from autofit import mock as m
from autofit.non_linear.analysis.combined import CombinedResult


class Analysis(af.Analysis):
    def log_likelihood_function(self, instance):
        return -random()


def test_integration():
    search = af.LBFGS(name="test_lbfgs")

    model = af.Collection(gaussian=af.Gaussian)

    n_analyses = 10

    analysis = Analysis()
    analysis = sum([analysis.with_model(model) for _ in range(n_analyses)])

    result = search.fit_sequential(model=model, analysis=analysis)

    assert len(os.listdir(Path(str(search.paths)).parent)) == n_analyses


def test_from_combined():
    combined_result = CombinedResult(
        [m.MockResult(model=af.Gaussian()) for _ in range(10)]
    )
