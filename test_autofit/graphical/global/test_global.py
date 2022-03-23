import pytest

import autofit as af
import autofit.graphical as g


def test_info(
        model_factor
):
    assert model_factor.global_prior_model.info == """PriorFactors

PriorFactor0 (AnalysisFactor0.one)                                                        UniformPrior, lower_limit = 0.0, upper_limit = 1.0

AnalysisFactors

AnalysisFactor0

one (PriorFactor0)                                                                        UniformPrior, lower_limit = 0.0, upper_limit = 1.0"""


def test_results(
        model_factor
):
    assert model_factor.graph.make_results_text(
        model_factor.global_prior_model
    ) == """PriorFactors

PriorFactor0 (AnalysisFactor0.one)                                                        0.5

AnalysisFactors

AnalysisFactor0

one (PriorFactor0)                                                                        0.5"""


class TestGlobalLikelihood:
    @pytest.mark.parametrize("unit_value, likelihood", [(0.5, 0.0), (0.0, -0.25)])
    def test_single_factor(self, model_factor, unit_value, likelihood):
        assert (
                model_factor.log_likelihood_function(
                    model_factor.global_prior_model.instance_from_unit_vector([unit_value])[0]
                )
                == likelihood
        )

    @pytest.mark.parametrize("unit_value, likelihood", [(0.5, 0.0), (0.0, -0.5)])
    def test_collection(self, model_factor, unit_value, likelihood):
        collection = g.FactorGraphModel(model_factor, model_factor)
        assert (
                collection.log_likelihood_function(
                    collection.global_prior_model.instance_from_unit_vector([unit_value])
                )
                == likelihood
        )

    @pytest.mark.parametrize(
        "unit_vector, likelihood", [([0.5, 0.0], 0.0), ([1.0, 0.5], -0.5)]
    )
    def test_two_factor(self, model_factor, model_factor_2, unit_vector, likelihood):
        collection = g.FactorGraphModel(model_factor, model_factor_2)

        assert (
                collection.log_likelihood_function(
                    collection.global_prior_model.instance_from_unit_vector(unit_vector)
                )
                == likelihood
        )

    def test_global_search(self, model_factor, model_factor_2):
        collection = g.FactorGraphModel(model_factor, model_factor_2)
        search = af.m.MockSearch()

        class Analysis(af.Analysis):
            def log_likelihood_function(self, instance):
                return collection.log_likelihood_function(instance)

        search.fit(collection.global_prior_model, Analysis())
