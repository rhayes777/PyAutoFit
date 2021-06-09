import autofit as af
from autofit import graphical as g
from autofit.mock.mock import MockAnalysis


class Analysis(MockAnalysis):
    def __init__(self):
        super().__init__()
        self.did_call_visualise = False

    def visualize(self, paths, instance, during_analysis):
        self.did_call_visualise = True


def test_visualize():
    analysis_0 = Analysis()

    gaussian_0 = af.Model(af.Gaussian)

    analysis_factor_0 = g.AnalysisFactor(
        prior_model=gaussian_0,
        analysis=analysis_0
    )

    factor_graph = g.FactorGraphModel(
        analysis_factor_0
    )

    model = factor_graph.global_prior_model
    instance = model.instance_from_prior_medians()

    factor_graph.visualize(
        af.DirectoryPaths(),
        instance,
        False
    )

    assert analysis_0.did_call_visualise is True
