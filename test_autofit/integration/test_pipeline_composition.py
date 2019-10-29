from autofit.optimize.non_linear.mock_nlo import MockNLO, MockAnalysis

from test_autofit import mock
import autofit as af


def make_pipeline_1(
        name,
        phase_folders,
        optimizer_class,
):
    phase = af.Phase(
        paths=af.Paths(
            phase_name="phase_1",
            phase_folders=phase_folders
        ),
        model=af.ModelMapper(
            one=af.PriorModel(
                mock.Galaxy
            )
        ),
        optimizer_class=optimizer_class,
        analysis_class=MockAnalysis
    )
    return af.Pipeline(f"{name}_1", phase)


def make_pipeline_2(
        name,
        phase_folders,
        optimizer_class,
):
    phase = af.Phase(
        paths=af.Paths(
            phase_name="phase_2",
            phase_folders=phase_folders
        ),
        model=af.ModelMapper(
            one=af.PriorModel(
                mock.Galaxy,
                redshift=af.last.variable.one.redshift
            )
        ),
        optimizer_class=optimizer_class,
        analysis_class=MockAnalysis
    )
    return af.Pipeline(f"{name}_2", phase)


def make_pipeline(
        name,
        phase_folders=tuple(),
        optimizer_class=MockNLO,
):
    return make_pipeline_1(
        name,
        phase_folders,
        optimizer_class,
    ) + make_pipeline_2(
        name,
        phase_folders,
        optimizer_class,
    )


def test_pipeline_composition():
    results = make_pipeline(
        "test"
    ).run(
        None
    )
    assert results[0].variable.one.redshift is results[1].variable.one.redshift
