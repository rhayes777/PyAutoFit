import os
import shutil
from os import path

import pytest

import autofit as af
from autoconf import conf
from autofit.mock import mock

directory = path.dirname(path.realpath(__file__))


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
        new_path=path.join(directory, "unit", "config"),
        output_path=path.join(directory, "output")
    )


@pytest.fixture(autouse=True)
def remove_output():
    try:
        shutil.rmtree(path.join(directory, "output"))
    except FileNotFoundError:
        pass


@pytest.fixture
def model():
    return af.ModelMapper()


@pytest.fixture(name="phase")
def make_phase():
    phase = af.AbstractPhase(search=af.MockSearch("phase name"))
    phase.model.one = af.PriorModel(mock.MockComponents, component=mock.MockClassx2)
    return phase


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

    collection.add("phase name", result)

    return collection
