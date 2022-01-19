import ast

import pytest
from astunparse import unparse

from autofit.tools.edenise import File
from autofit.tools.edenise import LineItem


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


def test_function(
        file
):
    string = """
def function():
    pass
        """
    item = LineItem.parse_fragment(
        string,
        parent=file
    )
    assert unparse(
        item.converted()
    ) == """

def function():
    pass
"""


def test_strip_type_annotations(
        file
):
    string = """
def function(argument: dict) -> Optional[Tuple[str, ...]]:
    pass
        """
    item = LineItem.parse_fragment(
        string,
        parent=file
    )
    assert unparse(
        item.converted()
    ) == """

def function(argument):
    pass
"""


def test_converted(file):
    converted_string = file.converted()
    print(converted_string)


def test_parse(
        examples_directory,
        eden_output_directory
):
    with open(examples_directory / "prior.py") as f:
        parsed = ast.parse(f.read())

    print(parsed)
