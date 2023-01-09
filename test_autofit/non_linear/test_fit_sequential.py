import os
from pathlib import Path
from random import random

import pytest

import autofit as af


class Analysis(af.Analysis):
    def log_likelihood_function(self, instance):
        return -random()


@pytest.fixture(name="search")
def make_search():
    return af.LBFGS(name="test_lbfgs")


@pytest.fixture(name="model")
def make_model():
    return af.Model(af.Gaussian)


@pytest.fixture(name="analysis")
def make_analysis():
    return Analysis()


def count_output(paths):
    return len(os.listdir(Path(str(paths)).parent))


def test_with_model(analysis, model, search):
    combined_analysis = sum([analysis.with_model(model) for _ in range(10)])

    result = search.fit_sequential(model=model, analysis=combined_analysis)

    assert count_output(search.paths) == 10
    assert len(result.child_results) == 10


@pytest.fixture(name="combined_analysis")
def make_combined_analysis(analysis):
    return sum([analysis for _ in range(10)])


def test_combined_analysis(combined_analysis, model, search):
    search.fit_sequential(model=model, analysis=combined_analysis)

    assert count_output(search.paths) == 10


def test_with_free_parameter(combined_analysis, model, search):
    combined_analysis = combined_analysis.with_free_parameters([model.centre])
    search.fit_sequential(
        model=model, analysis=combined_analysis,
    )

    assert count_output(search.paths) == 10
