import pytest

import autofit.mapper.prior_model.deferred
from autofit import exc
from autofit.mapper import model_mapper as mm
from autofit.mapper import prior as p
from autofit.mapper.prior_model import prior_model as pm
from test import mock
from test.mock import DeferredClass


@pytest.fixture(name="deferred_instance")
def make_deferred_instance():
    prior_model = pm.PriorModel(mock.SimpleClass)
    prior_model.two = autofit.mapper.prior_model.deferred.DeferredArgument()

    return prior_model.instance_for_arguments(
        {
            prior_model.one: 1.0
        }
    )


class TestCase:
    def test_is_deferred(self, deferred_instance):
        assert isinstance(deferred_instance,
                          autofit.mapper.prior_model.deferred.DeferredInstance)

    def test_instantiate(self, deferred_instance):
        instance = deferred_instance(two=2.0)
        assert isinstance(instance, mock.SimpleClass)
        assert instance.one == 1.0
        assert instance.two == 2.0

    def test_deferred_exception(self, deferred_instance):
        with pytest.raises(exc.DeferredInstanceException):
            print(deferred_instance.one)

    def test_deferred_config(self):
        mapper = mm.ModelMapper()
        mapper.DeferredClass = DeferredClass

        assert mapper.prior_count == 1

        deferred_instance = mapper.instance_from_unit_vector([1.0]).DeferredClass

        assert isinstance(deferred_instance,
                          autofit.mapper.prior_model.deferred.DeferredInstance)

        instance = deferred_instance(two=2.0)

        assert isinstance(instance, mock.DeferredClass)
        assert instance.one == 1.0
        assert instance.two == 2.0
