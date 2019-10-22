import numpy as np
import pytest

from test_autofit import mock


@pytest.fixture(name="source_light_profiles")
def make_source_light_profiles():
    source_light_profiles = mapper.CollectionPriorModel(light=mock.EllipticalLP)
    return source_light_profiles


@pytest.fixture(name="source")
def make_source(source_light_profiles):
    return mapper.PriorModel(mock.Galaxy, light_profiles=source_light_profiles)


@pytest.fixture(name="tracer_prior_model")
def make_tracer_prior_model(source):
    lens = mapper.PriorModel(
        mock.Galaxy,
        light_profiles=mapper.CollectionPriorModel(light=mock.EllipticalLP),
        mass_profiles=mapper.CollectionPriorModel(mass=mock.EllipticalMassProfile),
    )

    return mapper.PriorModel(mock.Tracer, lens_galaxy=lens, source_galaxy=source)


class TestCase(object):
    def test_simple_collection(self, source_light_profiles):
        assert len(source_light_profiles) == 1
        assert source_light_profiles.prior_count == 4

        instance = source_light_profiles.instance_for_arguments(
            {
                source_light_profiles.light.centre_0: 0.5,
                source_light_profiles.light.centre_1: 0.5,
                source_light_profiles.light.axis_ratio: 0.5,
                source_light_profiles.light.phi: 0.5,
            }
        )

        assert instance.light.centre[0] == 0.5

    def test_simple_model(self, source):
        assert source.prior_count == 5

        instance = source.instance_for_arguments(
            {
                source.light_profiles.light.centre_0: 0.5,
                source.light_profiles.light.centre_1: 0.5,
                source.light_profiles.light.axis_ratio: 0.5,
                source.light_profiles.light.phi: 0.5,
                source.redshift: 0.5,
            }
        )

        assert instance.light_profiles.light.centre[0] == 0.5

    def test_integration(self, tracer_prior_model):
        assert tracer_prior_model.prior_count == 14

        model_mapper = mapper.ModelMapper()
        model_mapper.Tracer = tracer_prior_model

        assert model_mapper.prior_count == 14

        prior = tracer_prior_model.source_galaxy.light_profiles.light.axis_ratio
        tracer_prior_model.lens_galaxy.mass_profiles.mass.axis_ratio = prior

        assert model_mapper.prior_count == 13

        instance = model_mapper.instance_from_prior_medians()
        grid = np.ndarray([0])
        tracer = instance.Tracer(grid=grid)

        assert isinstance(tracer, mock.Tracer)
        assert tracer.lens_galaxy.light_profiles.light.axis_ratio == 1.0
        assert tracer.grid is grid
