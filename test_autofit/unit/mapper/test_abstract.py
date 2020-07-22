import autofit as af
from test_autofit import mock


class TestCase:
    def test_transfer_tuples(self):

        model = af.ModelMapper()
        instance = af.ModelInstance()

        model.profile = af.PriorModel(mock.MockClassx2Tuple)
        assert model.prior_count == 2

        result = model.copy_with_fixed_priors(instance)
        assert result.prior_count == 2

        instance.profile = mock.MockClassx2Tuple()

        result = model.copy_with_fixed_priors(instance)
        assert result.prior_count == 0
        assert result.profile.one_tuple == (0.0, 0.0)
        assert isinstance(result.profile, af.PriorModel)

        instance = result.instance_from_unit_vector([])
        assert result.profile.one_tuple == (0.0, 0.0)
        assert isinstance(instance.profile, mock.MockClassx2Tuple)
