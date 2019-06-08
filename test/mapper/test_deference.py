from autofit.mapper import prior as p
from autofit.mapper import prior_model as pm
from test import mock


class TestCase:
    def test_deferred_argument(self):
        prior_model = pm.PriorModel(mock.SimpleClass)

        instance = prior_model.instance_for_arguments(
            {
                prior_model.one: 1.0,
                prior_model.two: p.DeferredArgument()
            }
        )

        assert isinstance(instance, pm.DeferredInstance)
