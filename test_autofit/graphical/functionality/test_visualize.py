import autofit as af
from autofit import graphical as g


class Analysis(af.m.MockAnalysis):
    def __init__(self):
        super().__init__()
        self.did_call_visualise = False

    def visualize(self, paths, instance, during_analysis):
        self.did_call_visualise = True


def test_visualize():
    analysis = Analysis()

    gaussian = af.Model(af.Gaussian)

    analysis_factor = g.AnalysisFactor(
        prior_model=gaussian,
        analysis=analysis
    )

    factor_graph = g.FactorGraphModel(
        analysis_factor
    )

    model = factor_graph.global_prior_model
    instance = model.instance_from_prior_medians()

    factor_graph.visualize(
        af.DirectoryPaths(),
        instance,
        False
    )

    assert analysis.did_call_visualise is True
