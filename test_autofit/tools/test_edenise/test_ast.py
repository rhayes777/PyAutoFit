import ast
import pytest

from autofit.tools.edenise import File


@pytest.fixture(
    name="parsed"
)
def make_parsed(
        examples_directory
):
    with open(examples_directory / "prior.py") as f:
        return ast.parse(f.read())


@pytest.fixture(
    name="file"
)
def make_file(
        examples_directory,
        package

):
    return File(
        examples_directory / "prior.py",
        prefix="PREFIX",
        parent=package
    )


def test_converted(file):
    converted_string = file.converted()
    print(converted_string)


class Import:
    def __init__(self, ast_import):
        self.ast_import = ast_import


def test_import(
        parsed
):
    import_ = Import(
        parsed.body[0]
    )
    assert import_.target_string == ""


def test_parse(
        examples_directory,
        eden_output_directory
):
    with open(examples_directory / "prior.py") as f:
        parsed = ast.parse(f.read())

    print(parsed)
