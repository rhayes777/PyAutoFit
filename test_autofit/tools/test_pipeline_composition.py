from os import path
import shutil

import autofit as af
from autoconf import conf
from autofit.mock.mock_search import MockSearch, MockAnalysis, MockSamples
from autofit.mock import mock

directory = path.dirname(path.realpath(__file__))

conf.instance = conf.Config(path.join(directory, "files", "config"))


def make_pipeline_1(name):

    if path.exists(path.join(conf.instance.output_path, "phase_1")):
        shutil.rmtree(path.join(conf.instance.output_path, "phase_1"))

    search = MockSearch(
        name="phase_1", samples=MockSamples(gaussian_tuples=[(0.5, 0.5)])
    )
    phase = af.Phase(
        model=af.PriorModel(mock.MockComponents, parameter=af.GaussianPrior(10.0, 1.0)),
        search=search,
        analysis_class=MockAnalysis,
    )
    return af.Pipeline(f"{name}_1", "", None, phase)


def make_pipeline_2(name):

    if path.exists(path.join(conf.instance.output_path, "phase_2")):
        shutil.rmtree(path.join(conf.instance.output_path, "phase_2"))

    search = MockSearch(
        name="phase_2", samples=MockSamples(gaussian_tuples=[(0.5, 0.5)])
    )
    phase = af.Phase(
        model=af.PriorModel(mock.MockComponents, parameter=af.last.model.parameter),
        search=search,
        analysis_class=MockAnalysis,
    )
    return af.Pipeline(f"{name}_2", "", None, phase)


def make_pipeline(name,):
    pipeline_2 = make_pipeline_2(name)
    pipeline_1 = make_pipeline_1(name)

    return pipeline_1 + pipeline_2


def test_pipeline_composition():

    pipeline = make_pipeline("test")
    results = pipeline.run(mock.MockDataset())
    assert results[0].model.parameter.mean == results[1].model.parameter.mean
