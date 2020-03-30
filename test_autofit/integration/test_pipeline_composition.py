import autofit as af
from autofit.optimize.non_linear.mock_nlo import MockNLO, MockAnalysis
from test_autofit import mock


def make_pipeline_1(name, phase_folders, non_linear_class):
    phase = af.Phase(
        phase_name="phase_1",
        phase_folders=phase_folders,
        model=af.PriorModel(
            mock.Galaxy,
            redshift=af.GaussianPrior(10.0, 1.0)
        ),
        non_linear_class=non_linear_class,
        analysis_class=MockAnalysis,
    )
    return af.Pipeline(f"{name}_1", phase)


def make_pipeline_2(name, phase_folders, non_linear_class):
    phase = af.Phase(
        phase_name="phase_2",
        phase_folders=phase_folders,
        model=af.PriorModel(
            mock.Galaxy,
            redshift=af.last.model.redshift
        ),
        non_linear_class=non_linear_class,
        analysis_class=MockAnalysis,
    )
    return af.Pipeline(f"{name}_2", phase)


def make_pipeline(
        name,
        phase_folders=tuple(),
        non_linear_class=MockNLO
):
    return make_pipeline_1(
        name,
        phase_folders,
        non_linear_class
    ) + make_pipeline_2(
        name,
        phase_folders,
        non_linear_class
    )


def test_pipeline_composition():
    pipeline = make_pipeline("test")
    results = pipeline.run(
        mock.MockDataset()
    )
    assert results[0].model.redshift.mean == results[1].model.redshift.mean
