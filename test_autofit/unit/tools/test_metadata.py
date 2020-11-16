import pytest

import autofit as af
from autofit.mock import mock


@pytest.fixture(name="phase")
def make_phase():
    return af.AbstractPhase(
        search=mock.MockSearch(paths=af.Paths(name="name", tag="phase_tag"))
    )


def test_metadata_dictionary(phase):
    phase.pipeline_name = "pipeline_name"
    phase.pipeline_tag = "pipeline_tag"
    assert phase._default_metadata == {
        "phase": "name",
        "phase_tag": "phase_tag",
        "pipeline": "pipeline_name",
        "pipeline_tag": "pipeline_tag",
    }
