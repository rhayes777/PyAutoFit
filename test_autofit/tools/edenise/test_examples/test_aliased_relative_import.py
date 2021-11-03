import pytest

from autofit.tools.edenise import File, Import


@pytest.fixture(
    name="file"
)
def make_file(
        package,
        examples_directory
):
    return File(
        examples_directory / "aliased_relative_import.py",
        parent=package,
        prefix=""
    )


def test_non_aliased(
        file
):
    assert Import.parse_fragment(
        "from .non_linear.grid.grid_search import GridSearch",
        parent=file
    ).target_string == "\nfrom VIS_CTI_Autofit.VIS_CTI_NonLinear.VIS_CTI_Grid.VIS_CTI_GridSearch import GridSearch\n"


def test_aliased_relative_import(
        file
):
    assert file.target_string == (
        "\nfrom VIS_CTI_Autofit.VIS_CTI_NonLinear.VIS_CTI_Grid.VIS_CTI_GridSearch import GridSearch\n"
    )
