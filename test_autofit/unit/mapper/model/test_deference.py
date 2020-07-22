import pytest

import autofit as af
from test_autofit import mock


@pytest.fixture(name="deferred_instance")
def make_deferred_instance():
    prior_model = af.PriorModel(mock.MockClassx2)
    prior_model.two = af.DeferredArgument()

    return prior_model.instance_for_arguments({prior_model.one: 1.0})


class TestCase:
    def test_is_deferred(self, deferred_instance):
        assert isinstance(deferred_instance, af.DeferredInstance)

    def test_instantiate(self, deferred_instance):
        instance = deferred_instance(two=2.0)
        assert isinstance(instance, mock.MockClassx2)
        assert instance.one == 1.0
        assert instance.two == 2.0

    def test_deferred_exception(self, deferred_instance):
        with pytest.raises(af.exc.DeferredInstanceException):
            print(deferred_instance.one)

    def test_deferred_config(self):
        mapper = af.ModelMapper()
        mapper.DeferredClass = mock.DeferredClass

        assert mapper.prior_count == 1

        deferred_instance = mapper.instance_from_unit_vector([1.0]).DeferredClass

        assert isinstance(deferred_instance, af.DeferredInstance)

        instance = deferred_instance(two=2.0)

        assert isinstance(instance, mock.DeferredClass)
        assert instance.one == 1.0
        assert instance.two == 2.0
