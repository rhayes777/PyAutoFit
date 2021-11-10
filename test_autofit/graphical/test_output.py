import autofit as af
from autoconf.conf import with_config
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
        super().fit(
            model,
            analysis,
        )
        return MockResult(model)


@with_config(
    "general",
    "output",
    "remove_files",
    value=False
)
def test_output(
        output_directory
):
    model_factor_1 = g.AnalysisFactor(
        af.Collection(
            one=af.UniformPrior()
        ),
        MockAnalysis(),
        name="factor_1"
    )
    model_factor_2 = g.AnalysisFactor(
        af.Collection(
            one=af.UniformPrior()
        ),
        MockAnalysis(),
        name="factor_2"
    )

    collection = g.FactorGraphModel(
        model_factor_1,
        model_factor_2
    )
    collection.optimise(
        MockSearch()
    )

    assert (output_directory / "factor_1").exists()
    assert (output_directory / "factor_2").exists()
