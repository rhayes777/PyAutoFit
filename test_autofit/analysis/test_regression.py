import pickle

from autofit.non_linear.analysis import CombinedAnalysis


def test_pickle(Analysis):
    analysis = Analysis() + Analysis()
    loaded = pickle.loads(
        pickle.dumps(
            analysis
        )
    )
    assert isinstance(
        loaded,
        CombinedAnalysis
    )
