import pytest

import autofit as af
from test_autofit import mock


@pytest.fixture(name="phase")
def make_phase():
    return af.AbstractPhase(phase_name="phase name")


class TestIteration:
    def test_index_promise(self, phase):
        model = af.Mapper()

        prior_0 = af.UniformPrior()
        prior_1 = af.UniformPrior()

        model.collection = af.Collection([
            prior_0,
            prior_1
        ])

        phase.model = model

        promise_0 = phase.result.model.collection[0]
        promise_1 = phase.result.model.collection[1]

        assert isinstance(
            promise_0,
            af.Promise
        )
        assert isinstance(
            promise_1,
            af.Promise
        )

        collection = af.ResultsCollection()
        instance = af.Instance()
        instance.collection = [1, 2]

        result = mock.Result(
            model=model,
            instance=instance
        )

        collection.add("phase name", result)

        prior = promise_0.populate(
            collection
        )
        assert prior == prior_0

        prior = promise_1.populate(
            collection
        )
        assert prior == prior_1
