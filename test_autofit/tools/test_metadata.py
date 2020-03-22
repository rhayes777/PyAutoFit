import pytest

import autofit as af


@pytest.fixture(
    name="phase"
)
def make_phase():
    return af.AbstractPhase(
        phase_name="phase_name",
        phase_tag="phase_tag"
    )


def test_metadata_dictionary(phase):
    phase.pipeline_name = "pipeline_name"
    phase.pipeline_tag = "pipeline_tag"
    assert phase._default_metadata == {
        "phase": "phase_name",
        "phase_tag": "phase_tag",
        "pipeline": "pipeline_name",
        "pipeline_tag": "pipeline_tag",
    }


class MockData:
    def __init__(self, name, metadata):
        self.name = name
        self.metadata = metadata


def test_metadata_text(phase):
    text = phase.make_metadata_text(
        MockData(
            "data",
            {
                "some": "metadata",
                "number": 1.0
            }
        )
    )
    assert text == """phase=phase_name
phase_tag=phase_tag
pipeline=
pipeline_tag=
some=metadata
number=1.0
dataset_name=data"""
