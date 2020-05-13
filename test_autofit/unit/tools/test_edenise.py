import pytest

from autofit.tools.edenise import Line, Converter


@pytest.fixture(
    name="as_line"
)
def make_as_line():
    return Line(
        "from .mapper.model import ModelInstance as Instance"
    )


@pytest.fixture(
    name="line"
)
def make_line():
    return Line(
        "from .mapper.model import ModelInstance"
    )


class Test:
    def test_line_is_import(self, as_line):
        assert as_line.is_import
        assert not Line(".mapper.model").is_import

    def test_source(self, as_line, line):
        assert as_line.source == "Instance"
        assert line.source == "ModelInstance"

    def test_target(self, as_line, line):
        assert as_line.target == "mapper.model.ModelInstance"
        assert line.target == "mapper.model.ModelInstance"

    def test_phase_property_line(self):
        line = Line(
            "from .tools.phase_property import PhaseProperty"
        )
        assert line.source == "PhaseProperty"
        assert line.target == "tools.phase_property.PhaseProperty"

    def test_replace(self, as_line, line):
        converter = Converter(
            "af",
            [as_line, line]
        )
        assert converter.convert(
            "af.ModelInstance\naf.Instance"
        ) == "af.mapper.model.ModelInstance\naf.mapper.model.ModelInstance"
