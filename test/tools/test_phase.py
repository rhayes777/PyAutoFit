import autofit as af
from test import mock
import pytest


@pytest.fixture(name="phase")
def make_phase():
    phase = af.AbstractPhase("phase name")
    phase.variable.one = af.PriorModel(mock.Galaxy)
    return phase


class TestCase:
    def test_variable_promise(self, phase):
        promise = phase.result.variable.one.redshift

        assert isinstance(promise, af.Promise)
        assert promise.path == ("one", "redshift")
        assert promise.is_constant is False
        assert promise.phase is phase

    def test_constant_promise(self, phase):
        promise = phase.result.constant.one.redshift

        assert isinstance(promise, af.Promise)
        assert promise.path == ("one", "redshift")
        assert promise.is_constant is True
        assert promise.phase is phase

    def test_non_existent(self, phase):
        with pytest.raises(AttributeError):
            assert phase.result.variable.one.bad

        with pytest.raises(AttributeError):
            assert phase.result.constant.one.bad
