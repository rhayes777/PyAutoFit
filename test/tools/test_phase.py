import pytest

import autofit as af
from test import mock


@pytest.fixture(name="phase")
def make_phase():
    phase = af.AbstractPhase("phase name")
    phase.variable.one = af.PriorModel(mock.Galaxy)
    return phase


@pytest.fixture(name="variable_promise")
def make_variable_promise(phase):
    return phase.result.variable.one.redshift


@pytest.fixture(name="constant_promise")
def make_constant_promise(phase):
    return phase.result.constant.one.redshift


@pytest.fixture(name="collection")
def make_collection():
    collection = af.ResultsCollection()
    variable = af.ModelMapper()
    variable.one = af.PriorModel(mock.Galaxy)
    constant = af.ModelMapper()
    constant.one = mock.Galaxy()

    collection.add(
        "phase name",
        mock.Result(
            variable=variable,
            constant=constant
        )
    )

    return collection


class TestCase:
    def test_variable_promise(self, variable_promise, phase):
        assert isinstance(variable_promise, af.Promise)
        assert variable_promise.path == ("one", "redshift")
        assert variable_promise.is_constant is False
        assert variable_promise.phase is phase

    def test_constant_promise(self, constant_promise, phase):
        assert isinstance(constant_promise, af.Promise)
        assert constant_promise.path == ("one", "redshift")
        assert constant_promise.is_constant is True
        assert constant_promise.phase is phase

    def test_non_existent(self, phase):
        with pytest.raises(AttributeError):
            assert phase.result.variable.one.bad

        with pytest.raises(AttributeError):
            assert phase.result.constant.one.bad

    def test_recover_variable(self, collection, variable_promise):
        result = variable_promise.populate(
            collection
        )

        assert result is collection[0].variable.one.redshift

    def test_recover_constant(self, collection, constant_promise):
        result = constant_promise.populate(
            collection
        )

        assert result is collection[0].constant.one.redshift

    def test_populate_prior_model_variable(self, collection, variable_promise):
        new_galaxy = af.PriorModel(
            mock.Galaxy,
            redshift=variable_promise
        )

        result = new_galaxy.populate(collection)

        assert result.redshift is collection[0].variable.one.redshift

    def test_populate_prior_model_constant(self, collection, constant_promise):
        new_galaxy = af.PriorModel(
            mock.Galaxy,
            redshift=constant_promise
        )

        result = new_galaxy.populate(collection)

        assert result.redshift is collection[0].constant.one.redshift
