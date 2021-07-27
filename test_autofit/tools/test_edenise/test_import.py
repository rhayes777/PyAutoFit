import pytest

from autofit.tools.edenise import Package, Import, LineItem


@pytest.fixture(
    name="import_"
)
def make_import(package):
    return Import(
        "from autofit.tools.edenise import Line",
        parent=package
    )


@pytest.fixture(
    name="local_import"
)
def make_local_import(
        package
):
    return LineItem(
        "    from autofit.tools.edenise import Line",
        parent=package
    )


def test_local_import_type(
        local_import
):
    assert isinstance(
        local_import,
        Import
    )


def test_local_import_target(
        local_import
):
    string = "    from VIS_CTI_Autofit.VIS_CTI_Tools.VIS_CTI_Edenise import Line"
    assert local_import.target_import_string == string


def test_project_import(import_, package):
    import_.parent = package["tools"]["edenise"]["converter"]
    assert "autofit/tools/edenise" in str(import_.path)
    assert import_.is_in_project is True


def test_non_project_import(package):
    import_ = Import(
        "import os",
        parent=package
    )
    import_.parent = package
    assert import_.is_in_project is False


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


def test_module_import_name(
        package
):
    conf = package["conf"]
    assert conf.target_import_string == "VIS_CTI_Conf as conf"
    assert conf.target_file_name == "VIS_CTI_Conf.py"

    package._should_rename_modules = False
    assert conf.target_import_string == "conf"
    assert conf.target_file_name == "conf.py"


def test_module_path_import_name(
        package
):
    package._should_rename_modules = False
    import_ = Import(
        "from autofit.non_linear.samples import NestSamples, Sample",
        parent=package
    )
    assert import_.target_string == "from VIS_CTI_Autofit.VIS_CTI_NonLinear.VIS_CTI_Samples import NestSamples, Sample"
