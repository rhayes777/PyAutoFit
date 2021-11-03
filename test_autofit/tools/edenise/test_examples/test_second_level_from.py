import pytest

from autofit.tools.edenise import Import, File


@pytest.fixture(
    name="file"
)
def make_file(
        package
):
    return package["database"]["aggregator"]["aggregator"]


@pytest.fixture(
    name="import_string"
)
def make_import_string():
    return "from VIS_CTI_Autofit.VIS_CTI_Database.VIS_CTI_Query.VIS_CTI_Query import AbstractQuery, Attribute"


def test_second_level_from(
        file,
        import_string
):
    assert Import.parse_fragment(
        "from ..query.query import AbstractQuery, Attribute",
        parent=file
    ).target_string.strip(" \n") == (
               import_string
           )


def test_string_in_file(
        file,
        import_string
):
    assert import_string in file.target_string


def test_relative_aliased_import(
        file
):
    string = "from VIS_CTI_Autofit.VIS_CTI_Database import VIS_CTI_Model as m"
    assert Import.parse_fragment(
        "from .. import model as m",
        parent=file
    ).target_string.strip(
        " \n"
    ) == string


def test_in_situ(
        package,
        examples_directory
):
    file = File(
        examples_directory / "aliased_relative_import_in_situ.py",
        parent=package["database"]["aggregator"],
        prefix=""
    )
    assert file.target_string == """
from VIS_CTI_Autofit.VIS_CTI_Database import VIS_CTI_Query
from VIS_CTI_Autofit.VIS_CTI_Database import VIS_CTI_Model
"""
