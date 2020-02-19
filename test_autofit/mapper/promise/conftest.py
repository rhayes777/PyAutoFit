import pytest

import autofit as af
from test_autofit import mock


@pytest.fixture(name="phase")
def make_phase():
    phase = af.AbstractPhase(phase_name="phase name")
    phase.model.one = af.PriorModel(mock.Galaxy, light=mock.EllipticalLP)
    return phase


@pytest.fixture(name="collection")
def make_collection():
    collection = af.ResultsCollection()
    model = af.ModelMapper()
    model.one = af.PriorModel(mock.Galaxy, light=mock.EllipticalLP)
    instance = af.ModelInstance()
    instance.one = mock.Galaxy(light=mock.EllipticalLP())

    result = mock.Result(model=model, instance=instance)

    model = af.ModelMapper()
    instance = af.ModelInstance()

    model.hyper_galaxy = mock.HyperGalaxy
    instance.hyper_galaxy = mock.HyperGalaxy()

    hyper_result = mock.Result(model=model, instance=instance)

    result.hyper_result = hyper_result

    collection.add("phase name", result)

    return collection
