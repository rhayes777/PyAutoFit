import os
from os import path
from matplotlib import pyplot

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import autofit as af
from autoconf import conf
from autofit import database as db
from autofit.mock import mock
from autofit.non_linear import samples as samp

directory = path.dirname(path.realpath(__file__))


class PlotPatch:
    def __init__(self):
        self.paths = []

    def __call__(self, path, *args, **kwargs):
        self.paths.append(path)


@pytest.fixture(name="plot_patch")
def make_plot_patch(monkeypatch):
    plot_patch = PlotPatch()
    monkeypatch.setattr(pyplot, "savefig", plot_patch)
    return plot_patch

@pytest.fixture(name="session")
def make_session():
    engine = create_engine('sqlite://')
    session = sessionmaker(bind=engine)()
    db.Base.metadata.create_all(engine)
    yield session
    session.close()
    engine.dispose()


@pytest.fixture(autouse=True)
def remove_reports():
    yield
    for d, _, files in os.walk(directory):
        for file in files:
            if file == "report.log":
                os.remove(path.join(d, file))


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance.push(
        new_path=path.join(directory, "config"),
        output_path=path.join(directory, "output")
    )


@pytest.fixture
def model():
    return af.ModelMapper()


@pytest.fixture(name="samples")
def make_samples():

    sample_0 = samp.Sample(log_likelihood=1.0, log_prior=2.0, weight=0.25)
    sample_1 = samp.Sample(log_likelihood=3.0, log_prior=5.0, weight=0.75)

    return af.StoredSamples(
        model=af.Mapper(),
        samples=[sample_0, sample_1],
    )

@pytest.fixture(name="result")
def make_result():
    model = af.Mapper()
    model.one = af.PriorModel(mock.MockComponents, component=mock.MockClassx2)
    return af.Result(model=model, samples=mock.MockSamples(), search=mock.MockSearch())


@pytest.fixture(name="collection")
def make_collection():
    collection = af.ResultsCollection()
    model = af.ModelMapper()
    model.one = af.PriorModel(mock.MockComponents, component=mock.MockClassx2)
    instance = af.ModelInstance()
    instance.one = mock.MockComponents(component=mock.MockClassx2())

    result = af.MockResult(model=model, instance=instance)

    model = af.ModelMapper()
    instance = af.ModelInstance()

    model.hyper_galaxy = mock.HyperGalaxy
    instance.hyper_galaxy = mock.HyperGalaxy()

    hyper_result = af.MockResult(model=model, instance=instance)

    result.hyper_result = hyper_result

    collection.add("search name", result)

    return collection
