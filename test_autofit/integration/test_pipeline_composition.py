from test_autofit import mock
from test_autofit.integration.tests import runner
import autofit as af
import sys

test_type = "lens__source"
test_name = "lens_light_mass__source__hyper_bg"
data_type = "lens_light__source_smooth"
data_resolution = "LSST"


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
        phase_folders,
        optimizer_class,
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
    results = runner.run_a_mock(
        sys.modules[__name__]
    )
    assert results[0].variable.one.redshift is results[1].variable.one.redshift
