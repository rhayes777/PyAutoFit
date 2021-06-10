from pathlib import Path

import pytest

from autofit.tools.edenise import Line, Converter, Package, File, Import


@pytest.fixture(name="as_line")
def make_as_line():
    return Line("from .mapper.model import ModelInstance as Instance")


@pytest.fixture(name="line")
def make_line():
    return Line("from .mapper.model import ModelInstance")


top_level_path = "top_level/path"


@pytest.fixture(
    name="package"
)
def make_package():
    directory = Path(
        __file__
    ).parent.parent.parent / "autofit"
    return Package.from_directory(
        directory,
        prefix="VIS_CTI"
    )


@pytest.fixture(
    name="child"
)
def make_child(package):
    return package.children[0]


class TestPackageStructure:
    def test_top_level(self, package):
        assert package.is_top_level is True
        assert len(package.children) > 1

    def test_child(self, child):
        assert isinstance(
            child,
            Package
        )
        assert child.is_top_level is False

    def test_path(self, package, child):
        assert package.target_name == "VIS_CTI_Autofit"
        assert str(package.target_path) == "VIS_CTI_Autofit"
        assert str(child.target_path) == f"VIS_CTI_Autofit/{child.target_name}"

    def test_import(self, package):
        import_ = Import("from autofit.tools.edenise import Line")
        import_.parent = package
        assert "autofit/tools/edenise.py" in str(import_.path)
        assert import_.is_in_project is True

        import_ = Import("import os")
        import_.parent = package
        assert import_.is_in_project is False

    def test_file(self, package):
        file = File(
            Path(__file__)
        )

        file.parent = package

        assert len(file.imports) > 1
        assert len(file.project_imports) == 1


class TestTestImports:
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

    def test_replace(self, as_line, line):
        converter = Converter("testfit", "tf", [as_line, line])
        assert (
                converter.convert("import testfit as tf\n\ntf.ModelInstance\ntf.Instance")
                == "from testfit.mapper.model import ModelInstance\n\n\nModelInstance\nModelInstance"
        )

    def test_replace_dotted(self):
        line = Line("from .text import formatter")
        assert line.joint_source == "text.formatter"
        converter = Converter("testfit", "tf", [line])
        assert (
                converter.convert(
                    "import testfit as tf\n\ntf.text.formatter.label_and_label_string"
                )
                == "from testfit.text import formatter\n\n\nformatter.label_and_label_string"
        )


def test_convert_formatter():
    unit_test_directory = Path(__file__).parent.parent
    test_path = unit_test_directory / "text/test_samples_text.py"
    with open(test_path) as f:
        string = f.read()
    converter = Converter.from_prefix_and_source_directory(
        "autofit", "af", unit_test_directory.parent / "autofit"
    )
    result = converter.convert(string)

    assert "from autofit" in result
