import autofit as af
from autofit import graphical as g
from autofit.mock.mock import MockAnalysis


class MockResult(af.MockResult):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def projected_model(self):
        return self.model


class MockSearch(af.MockSearch):
    def fit(
            self,
            model,
            analysis,
            info=None,
            pickle_files=None,
            log_likelihood_cap=None
    ):
        return MockResult(model)


def test_output():
    model_factor_1 = g.AnalysisFactor(
        af.Collection(
            one=af.UniformPrior()
        ),
        MockAnalysis()
    )
    model_factor_2 = g.AnalysisFactor(
        af.Collection(
            one=af.UniformPrior()
        ),
        MockAnalysis()
    )

    collection = g.FactorGraphModel(
        model_factor_1,
        model_factor_2
    )
    collection.optimise(
        MockSearch()
    )
