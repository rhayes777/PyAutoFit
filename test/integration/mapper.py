from autofit import mapper
from test import mock


class TestCase(object):
    def test(self):
        source = mapper.PriorModel(
            mock.Galaxy,
            light_profiles=mapper.CollectionPriorModel(
                dict(
                    light=mock.EllipticalLP
                )
            )
        )

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

        model_mapper = mapper.ModelMapper()

        model_mapper.tracer = mapper.PriorModel(
            mock.Tracer,
            lens=lens,
            source=source
        )

        assert model_mapper.prior_count == 14
