from test_autofit import mock
import autofit as af

from test_autolens.mock.mock_pipeline import MockNLO


def make_pipeline_1(
        name,
        phase_folders,
        optimizer_class,
):
    phase = af.AbstractPhase(
        paths=af.Paths(
            phase_name="phase_1",
            phase_folders=phase_folders
        ),
        model=af.ModelMapper(
            one=af.PriorModel(
                mock.Galaxy
            )
        ),
        optimizer_class=optimizer_class
    )
    return af.Pipeline(f"{name}_1", phase)


def make_pipeline_2(
        name,
        phase_folders,
        optimizer_class,
):
    phase = af.AbstractPhase(
        paths=af.Paths(
            phase_name="phase_2",
            phase_folders=phase_folders
        ),
        model=af.ModelMapper(
            one=af.PriorModel(
                mock.Galaxy,
                redshift=af.last.one.redshift
            )
        ),
        optimizer_class=optimizer_class
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
    )
    assert results[0].variable.one.redshift is results[1].variable.one.redshift
