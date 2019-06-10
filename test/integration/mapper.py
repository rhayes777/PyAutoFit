from autofit import mapper
from test import mock


class TestCase(object):
    def test(self):
        source_light_profiles = mapper.CollectionPriorModel(
            dict(
                light=mock.EllipticalLP
            )
        )
        assert len(source_light_profiles) == 1
        assert source_light_profiles.prior_count == 4

        source = mapper.PriorModel(
            mock.Galaxy,
            light_profiles=source_light_profiles
        )

        assert source.prior_count == 5

        lens = mapper.PriorModel(
            mock.Galaxy,
            light_profiles=mapper.CollectionPriorModel(
                dict(
                    light=mock.EllipticalLP
                )
            ),
            mass_profiles=mapper.CollectionPriorModel(
                dict(
                    light=mock.EllipticalMassProfile
                )
            )
        )

        tracer = mapper.PriorModel(
            mock.Tracer,
            lens=lens,
            source=source
        )

        assert tracer.prior_count == 14

        model_mapper = mapper.ModelMapper()
        model_mapper.tracer = tracer

        assert model_mapper.prior_count == 14
