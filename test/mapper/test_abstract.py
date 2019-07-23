import autofit as af
from test import mock


class TestCase:
    def test_transfer_tuples(self):
        variable = af.ModelMapper()
        instance = af.ModelInstance()

        variable.profile = af.PriorModel(mock.GeometryProfile)
        assert variable.prior_count == 2

        result = variable.copy_with_fixed_priors(
            instance
        )
        assert result.prior_count == 2

        instance.profile = mock.GeometryProfile()

        result = variable.copy_with_fixed_priors(
            instance
        )
        assert result.prior_count == 0
