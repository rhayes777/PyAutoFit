from pathlib import Path

import pytest

from autofit.tools.edenise import Package, File, Import

package_directory = Path(
    __file__
).parent.parent.parent.parent / "autofit"


@pytest.fixture(
    name="package"
)
def make_package():
    return Package.from_directory(
        package_directory,
        prefix="VIS_CTI"
    )


@pytest.fixture(
    name="child"
)
def make_child(package):
    return package.children[0]


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
    assert str(package.target_path) == "VIS_CTI_Autofit"
    assert str(child.target_path) == f"VIS_CTI_Autofit/{child.target_name}"


@pytest.fixture(
    name="import_"
)
def make_import(package):
    import_ = Import("from autofit.tools.edenise import Line")
    import_.parent = package
    return import_


def test_project_import(import_):
    assert "autofit/tools/edenise/converter.py" in str(import_.path)
    assert import_.is_in_project is True


def test_non_project_import(package):
    import_ = Import("import os")
    import_.parent = package
    assert import_.is_in_project is False


def test_import_file(import_):
    assert isinstance(
        import_.file,
        File
    )


def test_target_import_string(import_):
    string = "from VIS_CTI_Autofit.VIS_CTI_Tools.VIS_CTI_Edenise import Line"
    assert import_.target_import_string == string


def test_multi_import(
        package
):
    import_ = Import(
        "from autofit.tools.edenise import Package, File, Import"
    )
    import_.parent = package
    string = "from VIS_CTI_Autofit.VIS_CTI_Tools.VIS_CTI_Edenise import Package, File, Import"
    assert import_.target_import_string == string


def test_file(package):
    file = File(
        Path(__file__),
        prefix=""
    )

    file.parent = package

    assert len(file.imports) > 1
    assert len(file.project_imports) == 1


def test_convert_file():
    pass
