import autofit as af


def test_metadata_dictionary():
    phase = af.AbstractPhase(
        phase_name="phase_name",
        phase_tag="phase_tag"
    )
    phase.pipeline_name = "pipeline_name"
    phase.pipeline_tag = "pipeline_tag"
    assert phase._default_metadata == {
        "phase": "phase_name",
        "phase_tag": "phase_tag",
        "pipeline": "pipeline_name",
        "pipeline_tag": "pipeline_tag",
    }
