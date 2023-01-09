from random import random

import autofit as af
from autofit import mock as m
from autofit.non_linear.analysis.combined import CombinedResult


class Analysis(af.Analysis):
    def log_likelihood_function(self, instance):
        return -random()


def test_integration():
    search = af.LBFGS()
    analysis = sum([Analysis() for _ in range(10)])

    model = af.Collection(gaussian=af.Gaussian)

    result = search.fit(model=model, analysis=analysis)

    print(result.model)


def test_from_combined():
    combined_result = CombinedResult(
        [m.MockResult(model=af.Gaussian()) for _ in range(10)]
    )
