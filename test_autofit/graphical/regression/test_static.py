from collections import Counter

import autofit as af


class StaticSearch(af.NonLinearSearch):
    def __init__(self):
        self.optimisation_counter = Counter()
        self._paths = af.DirectoryPaths()
        self.delta = 1.0

    def fit(
            self,
            model,
            analysis,
            info=None,
            pickle_files=None,
            log_likelihood_cap=None
    ):
        return StaticResult(model)

    def _fit(self, model, analysis, log_likelihood_cap=None):
        pass

    @property
    def config_type(self):
        pass

    @property
    def samples_cls(self):
        pass

    def samples_from(self, model):
        pass

    def samples_via_results_from(self, model):
        pass


class StaticResult:
    def __init__(self, model):
        self.projected_model = model


def test_static():
    analysis = af.mock.MockAnalysis()
    factor_graph_model = af.FactorGraphModel(
        af.AnalysisFactor(
            prior_model=af.Model(
                af.Gaussian,
                centre=af.GaussianPrior(mean=1.0, sigma=2.0),
                normalization=af.GaussianPrior(mean=1.0, sigma=2.0),
                sigma=af.GaussianPrior(mean=1.0, sigma=2.0),
            ),
            analysis=analysis
        )
    )
    original_mean_field = factor_graph_model.mean_field_approximation()
    factor = factor_graph_model.graph.factors[0]

    mean_field = original_mean_field

    search = StaticSearch()

    for i in range(3):
        mean_field, _ = search.optimise(
            model_approx=mean_field,
            factor=factor,
        )

        assert list(original_mean_field.mean_field.values()) == list(mean_field.mean_field.values())
