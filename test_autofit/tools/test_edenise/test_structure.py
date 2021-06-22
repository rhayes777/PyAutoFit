import shutil
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
    return Package(
        package_directory,
        prefix="VIS_CTI",
        is_top_level=True
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
    return Import(
        "from autofit.tools.edenise import Line",
        parent=package
    )


def test_project_import(import_, package):
    import_.parent = package["tools"]["edenise"]["converter"]
    assert "autofit/tools/edenise/converter.py" in str(import_.path)
    assert import_.is_in_project is True


def test_non_project_import(package):
    import_ = Import(
        "import os",
        parent=package
    )
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
        "from autofit.tools.edenise import Package, File, Import",
        parent=package
    )
    string = "from VIS_CTI_Autofit.VIS_CTI_Tools.VIS_CTI_Edenise import Package, File, Import"
    assert import_.target_import_string == string


def test_init(
        package
):
    assert package["__init__"].target_name == "__init__.py"


def test_import_as(
        package
):
    import_ = Import(
        "from autofit.tools import edenise as e",
        parent=package
    )
    string = "from VIS_CTI_Autofit.VIS_CTI_Tools import VIS_CTI_Edenise as e"
    assert import_.target_import_string == string


def test_package_import(
        package
):
    import_ = Import(
        "from autofit.tools import edenise",
        parent=package["tools"]
    )
    assert isinstance(
        package["tools"]["edenise"],
        Package
    )
    assert import_.module_path == ["autofit", "tools"]
    assert isinstance(
        import_.module["edenise"],
        Package
    )

    string = "from VIS_CTI_Autofit.VIS_CTI_Tools import VIS_CTI_Edenise"
    assert import_.target_import_string == string


@pytest.mark.parametrize(
    "string, result",
    [
        ("from . import util", "from autofit.tools import util"),
        ("from .. import conf", "from autofit import conf"),
        ("from ..tools import util", "from autofit.tools import util")
    ]
)
def test_relative_import(
        package,
        string,
        result
):
    import_ = Import(
        string,
        parent=package["tools"]["namer"]
    )

    assert import_.string == result


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
