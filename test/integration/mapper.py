import numpy as np
import pytest

from autofit import mapper
from test import mock


@pytest.fixture(name="source_light_profiles")
def make_source_light_profiles():
    source_light_profiles = mapper.CollectionPriorModel(
        light=mock.EllipticalLP
    )
    return source_light_profiles


@pytest.fixture(name="source")
def make_source(source_light_profiles):
    return mapper.PriorModel(
        mock.Galaxy,
        light_profiles=source_light_profiles
    )


@pytest.fixture(name="tracer_prior_model")
def make_tracer_prior_model(source):
    lens = mapper.PriorModel(
        mock.Galaxy,
        light_profiles=mapper.CollectionPriorModel(
            light=mock.EllipticalLP
        ),
        mass_profiles=mapper.CollectionPriorModel(
            light=mock.EllipticalMassProfile
        )
    )

    return mapper.PriorModel(
        mock.Tracer,
        lens_galaxy=lens,
        source_galaxy=source
    )


class TestCase(object):
    def test_simple_collection(self, source_light_profiles):
        assert len(source_light_profiles) == 1
        assert source_light_profiles.prior_count == 4

        instance = source_light_profiles.instance_for_arguments(
            source_light_profiles.light
        )

    def test_simple_model(self, source):
        assert source.prior_count == 5

    def test(self, tracer_prior_model):
        assert tracer_prior_model.prior_count == 14

        model_mapper = mapper.ModelMapper()
        model_mapper.Tracer = tracer_prior_model

        assert model_mapper.prior_count == 14

        instance = model_mapper.instance_from_prior_medians()
        grid = np.ndarray([0])
        tracer = instance.Tracer(grid=grid)

        assert isinstance(tracer, mock.Tracer)
        assert tracer.lens_galaxy.light_profiles.light.axis_ratio == 1.0
