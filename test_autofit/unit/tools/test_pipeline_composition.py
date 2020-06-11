from autofit.non_linear.mock.mock_nlo import MockSearch, MockAnalysis

from os import path
import autofit as af
from autoconf import conf
from test_autofit import mock


directory = path.dirname(path.realpath(__file__))

conf.instance = conf.Config(
    config_path=path.join(directory, "../config"),
)

def make_pipeline_1(name, phase_folders, search):
    phase = af.Phase(
        phase_name="phase_1",
        phase_folders=phase_folders,
        model=af.PriorModel(
            mock.Galaxy,
            redshift=af.GaussianPrior(10.0, 1.0)
        ),
        search=search,
        analysis_class=MockAnalysis,
    )
    return af.Pipeline(f"{name}_1", phase)


def make_pipeline_2(name, phase_folders, search):
    phase = af.Phase(
        phase_name="phase_2",
        phase_folders=phase_folders,
        model=af.PriorModel(
            mock.Galaxy,
            redshift=af.last.model.redshift
        ),
        search=search,
        analysis_class=MockAnalysis,
    )
    return af.Pipeline(f"{name}_2", phase)


def make_pipeline(
        name,
        phase_folders=tuple(),
        search=MockSearch()
):
    return make_pipeline_1(
        name,
        phase_folders,
        search
    ) + make_pipeline_2(
        name,
        phase_folders,
        search
    )


def test_pipeline_composition():
    pipeline = make_pipeline("test")
    results = pipeline.run(
        mock.MockDataset()
    )
    assert results[0].model.redshift.mean == results[1].model.redshift.mean
