import autofit as af
import autofit.tools.promise
from test import mock


class TestCase:
    def test_variable_promise(self):
        phase = af.AbstractPhase("phase name")
        phase.variable.one = af.PriorModel(mock.Galaxy)
        promise = phase.result.variable.one.redshift

        assert isinstance(promise, autofit.tools.promise.Promise)
        assert promise.path == ("one", "redshift")
        assert promise.is_constant is False
        assert promise.phase is phase
