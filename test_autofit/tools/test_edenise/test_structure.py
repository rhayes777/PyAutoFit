import shutil
from pathlib import Path

import pytest

from autofit.tools.edenise import Package, File


@pytest.fixture(
    name="child"
)
def make_child(package):
    return sorted(
        package.children,
        key=str
    )[1]


def test_top_level(package):
    assert package.is_top_level is True
    assert len(package.children) > 1


def test_child(child):
    assert isinstance(
        child,
        Package
    )
    assert child.is_top_level is False


def test_path(package, child):
    assert package.target_name == "VIS_CTI_Autofit"
    assert str(package.target_path) == "VIS_CTI/VIS_CTI_Autofit/python/VIS_CTI_Autofit"
    assert str(child.target_path) == f"VIS_CTI/VIS_CTI_Autofit/python/VIS_CTI_Autofit/{child.target_name}"


def test_init(
        package
):
    assert package["__init__"].target_file_name == "__init__.py"


def test_get_item(
        package
):
    assert isinstance(
        package[
            "mapper"
        ],
        Package
    )


def test_file(package):
    file = File(
        Path(__file__),
        prefix="",
        parent=package
    )

    assert len(file.imports) > 1
    assert len(file.project_imports) == 1


def test_generate_target_directories(
        output_path
):
    assert output_path.exists()
    assert (
            output_path / "VIS_CTI_Autofit/VIS_CTI_Tools/VIS_CTI_Edenise/VIS_CTI_Structure/VIS_CTI_Item.py"
    )


@pytest.fixture(
    name="output_path"
)
def make_output_path(package):
    output_directory = Path(
        __file__
    ).parent / "output"
    package.generate_target(
        output_directory
    )
    yield output_directory
    shutil.rmtree(output_directory)
