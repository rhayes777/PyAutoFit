import autofit as af
import autofit.tools.promise
from test import mock


class TestCase:
    def test_promise(self):
        phase = af.AbstractPhase("phase name")
        phase.variable.one = af.PriorModel(mock.Galaxy)
        assert isinstance(phase.result.variable.one.redshift, autofit.tools.promise.Promise)
