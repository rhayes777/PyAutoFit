import autofit as af
from autofit.non_linear.paths.null import NullPaths


def test_make_result():
    model = af.Model(af.Gaussian)
    factor_graph_model = af.FactorGraphModel(
        af.AnalysisFactor(
            model,
            af.Analysis(),
        )
    )
    result = factor_graph_model.make_result(
        samples_summary=af.SamplesSummary(
            max_log_likelihood_sample=af.Sample(
                0,
                0,
                0,
                kwargs={
                    ("0", "centre"): 1.0,
                    ("0", "normalization"): 1.0,
                    ("0", "sigma"): 1.0,
                },
            ),
            model=af.Collection(model),
        ),
        paths=NullPaths(),
    )
    assert len(result.child_results) == 1
    assert isinstance(result.model, af.Collection)

    (child_result,) = result.child_results
    assert child_result.model == model
