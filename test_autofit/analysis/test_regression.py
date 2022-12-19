import pickle

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
        self.is_modified = False

    def log_likelihood_function(self, instance):
        pass

    def make_result(self, samples, model, sigma=1.0, use_errors=True, use_widths=False):
        return MyResult(model=model, samples=samples)

    def modify_before_fit(self, paths, model):
        self.is_modified = True
        return self


def test_result_type():
    model = af.Model(af.Gaussian)

    analysis = MyAnalysis().with_model(model)

    result = analysis.make_result(None, model)

    assert isinstance(result, MyResult)


def test_combined_before_fit():
    analysis = MyAnalysis() + MyAnalysis()

    analysis = analysis.modify_before_fit(None, None)

    assert analysis[0].is_modified
