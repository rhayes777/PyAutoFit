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

    def test_recover_variable(self, variable_promise):
        collection = af.ResultsCollection()
        variable = af.ModelMapper()
        variable.one = af.PriorModel(mock.Galaxy)

        collection.add(
            "phase name",
            mock.Result(
                variable=variable
            )
        )

        result = variable_promise.populate(
            collection
        )

        assert result is variable.one.redshift
