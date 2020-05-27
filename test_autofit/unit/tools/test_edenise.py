from pathlib import Path

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
        assert as_line._target == "mapper.model.ModelInstance"
        assert line._target == "mapper.model.ModelInstance"

    def test_hash(self, as_line, line):
        assert hash(as_line) == hash(line)

    def test_phase_property_line(self):
        line = Line(
            "from .tools.phase_property import PhaseProperty"
        )
        assert line.source == "PhaseProperty"
        assert line._target == "tools.phase_property.PhaseProperty"

    def test_replace(self, as_line, line):
        converter = Converter(
            "testfit",
            "tf",
            [as_line, line]
        )
        assert converter.convert(
            "import testfit as tf\n\ntf.ModelInstance\ntf.Instance"
        ) == "from testfit.mapper.model import ModelInstance\n\n\nModelInstance\nModelInstance"

    def test_replace_dotted(self):
        line = Line(
            "from .text import formatter"
        )
        assert line.joint_source == "text.formatter"
        converter = Converter(
            "testfit",
            "tf",
            [line]
        )
        assert converter.convert(
            "import testfit as tf\n\ntf.text.formatter.label_and_label_string"
        ) == "from testfit.text import formatter\n\n\nformatter.label_and_label_string"


def test_convert_formatter():
    unit_test_directory = Path(__file__).parent.parent
    test_path = unit_test_directory / "text/test_model_text.py"
    with open(test_path) as f:
        string = f.read()
    converter = Converter.from_prefix_and_source_directory(
        "autofit",
        "af",
        unit_test_directory.parent.parent / "autofit"
    )
    result = converter.convert(string)
    print(result)
    assert "from autofit" in result
