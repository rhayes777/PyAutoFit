import pytest

import autofit as af


class MockPhaseOutput:
    def __init__(
            self,
            pipeline,
            phase,
            dataset
    ):
        self.pipeline = pipeline
        self.phase = phase
        self.dataset = dataset


@pytest.fixture(
    name="aggregator"
)
def make_aggregator():
    aggregator = af.Aggregator("")
    aggregator.phases = [
        MockPhaseOutput(
            "pipeline1",
            "phase1",
            "dataset1"
        ),
        MockPhaseOutput(
            "pipeline1",
            "phase2",
            "dataset1"
        ),
        MockPhaseOutput(
            "pipeline2",
            "phase2",
            "dataset2"
        )
    ]
    return aggregator


def test_attribute(aggregator):
    assert aggregator.pipeline == [
        "pipeline1",
        "pipeline1",
        "pipeline2"
    ]
    assert aggregator.phase == [
        "phase1",
        "phase2",
        "phase2"
    ]
    assert aggregator.dataset == [
        "dataset1",
        "dataset1",
        "dataset2"
    ]


def test_filter(aggregator):
    result = aggregator.filter(
        pipeline="pipeline1"
    )
    assert len(result) == 2
    assert result[0].pipeline == "pipeline1"

    result = aggregator.filter(
        pipeline="pipeline1",
        phase="phase1"
    )

    assert len(result) == 1
    assert result[0].pipeline == "pipeline1"

    result = aggregator.filter(
        pipeline="pipeline1"
    ).filter(
        phase="phase1"
    )

    assert len(result) == 1
    assert result[0].pipeline == "pipeline1"
