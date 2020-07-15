import pytest

import autofit as af
from test_autofit import mock


@pytest.fixture(
    name="phase"
)
def make_phase():
    return af.AbstractPhase(
        phase_name="phase_name",
        phase_tag="phase_tag",
        search=mock.MockSearch()
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


def test_datset_name_in_metadata_text(phase):
    text = phase.make_metadata_text(
        dataset_name="data"
    )
    print(text)
    assert text == """phase=phase_name
phase_tag=phase_tag
pipeline=
pipeline_tag=
dataset_name=data"""
