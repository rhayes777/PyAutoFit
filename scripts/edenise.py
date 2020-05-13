from pathlib import Path

import pytest
import re


class Line:
    def __init__(self, string):
        self.string = string

    @property
    def is_import(self):
        return self.string.startswith("from")

    @property
    def source(self):
        return re.match(".* as (.+)", self.string).group(1)

    @property
    def target(self):
        pass


class Converter:
    def __init__(self, source_directory):
        self.source_directory = source_directory
        with open(self._init_directory) as f:
            lines = map(Line, f.readlines())

    @property
    def _init_directory(self):
        return f"{self.source_directory}/__init__.py"


@pytest.fixture(
    name="as_line"
)
def make_as_line():
    return Line(
        "from .mapper.model import ModelInstance as Instance"
    )


class Test:
    def test_line_is_import(self, as_line):
        assert as_line.is_import
        assert not Line(".mapper.model").is_import

    def test_source(self, as_line):
        assert as_line.source == "Instance"
        # assert Line(
        #     "from .mapper.model import ModelInstance"
        # ).source == "ModelInstance"


if __name__ == "__main__":
    root_directory = Path(__file__).parent.parent

    converter = Converter(
        source_directory=root_directory / "autofit"
    )
