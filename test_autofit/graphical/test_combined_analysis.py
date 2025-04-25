import autofit as af
from autofit import AbstractPaths, DirectoryPaths, AbstractPriorModel
from autofit.non_linear.paths.null import NullPaths
from autofit.non_linear.paths.sub_directory_paths import SubDirectoryPaths


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


class TestAnalysis(af.Analysis):
    def __init__(self):
        super().__init__()
        calls = []
        self.calls = calls

        class Visualizer(af.Visualizer):
            @staticmethod
            def visualize_before_fit(
                analysis,
                paths: SubDirectoryPaths,
                model: AbstractPriorModel,
            ):
                calls.append(("visualize_before_fit", paths.analysis_name))

            @staticmethod
            def visualize(
                analysis,
                paths: SubDirectoryPaths,
                instance,
                during_analysis,
            ):
                calls.append(("visualize", paths.analysis_name))

        self.Visualizer = Visualizer

    def log_likelihood_function(self, instance):
        return 0.0

    def save_attributes(self, paths: SubDirectoryPaths):
        self.calls.append(("save_attributes", paths.analysis_name))

    def save_results(self, paths: SubDirectoryPaths, result):
        self.calls.append(("save_results", paths.analysis_name))


def test_visualize():
    model = af.Model(af.Gaussian)
    analysis = TestAnalysis()

    analysis_factor = af.AnalysisFactor(
        prior_model=model,
        analysis=analysis,
    )
    factor_graph = af.FactorGraphModel(analysis_factor)
    factor_graph.visualize(
        DirectoryPaths(),
        af.Collection(model).instance_from_prior_medians(),
        False,
    )

    assert analysis.calls == [
        ("visualize", "analyses/analysis_0"),
    ]
