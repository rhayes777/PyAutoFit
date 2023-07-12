import pickle

import pytest

import autofit as af
from autofit.non_linear.analysis import CombinedAnalysis


def test_pickle(Analysis):
    analysis = Analysis() + Analysis()
    loaded = pickle.loads(pickle.dumps(analysis))
    assert isinstance(loaded, CombinedAnalysis)


class MyResult(af.Result):
    pass


class MyAnalysis(af.Analysis):
    def __init__(self):
        self.is_modified_before = False
        self.is_modified_after = False

    def log_likelihood_function(self, instance):
        pass

    def make_result(self, samples, sigma=1.0, use_errors=True, use_widths=False):
        return MyResult(samples=samples)

    def modify_before_fit(self, paths, model):
        self.is_modified_before = True
        return self

    def modify_after_fit(self, paths, model, result):
        self.is_modified_after = True
        return self


def test_result_type():
    model = af.Model(af.Gaussian)

    analysis = MyAnalysis().with_model(model)

    result = analysis.make_result(None)

    assert isinstance(result, MyResult)


@pytest.fixture(name="combined_analysis")
def make_combined_analysis():
    return MyAnalysis() + MyAnalysis()


@pytest.fixture(name="paths")
def make_paths():
    return af.DirectoryPaths()


def test_combined_before_fit(combined_analysis, paths):
    combined_analysis = combined_analysis.modify_before_fit(paths, [None])

    assert combined_analysis[0].is_modified_before


def test_combined_after_fit(combined_analysis, paths):
    result = combined_analysis.make_result(None)

    combined_analysis = combined_analysis.modify_after_fit(paths, [None], result)

    assert combined_analysis[0].is_modified_after
