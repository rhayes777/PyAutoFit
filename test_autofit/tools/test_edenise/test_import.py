import ast

import pytest
from astunparse import unparse

from autofit.tools.edenise import Import


@pytest.fixture(
    name="import_"
)
def make_import(package):
    return Import(
        ast.parse(
            "from autofit.tools.edenise import Line"
        ).body[0],
        parent=package
    )


@pytest.mark.parametrize(
    "source, target",
    [
        ("import autofit.tools.edenise", "\nimport VIS_CTI_Autofit.VIS_CTI_Tools.VIS_CTI_Edenise\n"),
        ("import autofit", "\nimport VIS_CTI_Autofit\n"),
        ("import autofit.tools.edenise.Line", "\nimport VIS_CTI_Autofit.VIS_CTI_Tools.VIS_CTI_Edenise.Line\n"),
    ]
)
def test_direct_import(
        package,
        source,
        target
):
    import_ = Import(
        ast.parse(
            source
        ).body[0],
        parent=package
    )
    target_string = target
    assert unparse(import_.converted()) == target_string


def test_project_import(import_, package):
    import_.parent = package["tools"]["edenise"]["converter"]
    assert import_.is_in_project is True


def test_non_project_import(package):
    import_ = Import(
        ast.parse("import os").body[0],
        parent=package
    )
    import_.parent = package
    assert import_.is_in_project is False
    assert import_.target_string == """
import os
"""


def test_non_project_from_import(
        package
):
    import_ = Import(
        ast.parse("from hashlib import md5").body[0],
        parent=package
    )
    assert import_.target_string == """
from hashlib import md5
"""


def test_target_import_string(import_):
    string = "\nfrom VIS_CTI_Autofit.VIS_CTI_Tools.VIS_CTI_Edenise import Line\n"
    target_import_string = unparse(import_.converted())
    assert target_import_string == string


def test_multi_import(
        package
):
    import_ = Import(
        ast.parse(
            "from autofit.tools.edenise import Package, File, Import"
        ).body[0],
        parent=package
    )
    string = "\nfrom VIS_CTI_Autofit.VIS_CTI_Tools.VIS_CTI_Edenise import Package, File, Import\n"
    target_import_string = unparse(import_.converted())
    assert target_import_string == string


def test_import_as(
        package
):
    import_ = Import(
        ast.parse(
            "from autofit.tools import edenise as e"
        ).body[0],
        parent=package
    )
    string = "\nfrom VIS_CTI_Autofit.VIS_CTI_Tools import VIS_CTI_Edenise as e\n"
    target_import_string = unparse(import_.converted())
    assert target_import_string == string


def test_package_import(
        package
):
    import_ = Import(
        ast.parse(
            "from autofit.tools import edenise"
        ).body[0],
        parent=package["tools"]
    )

    string = "\nfrom VIS_CTI_Autofit.VIS_CTI_Tools import VIS_CTI_Edenise\n"
    target_import_string = unparse(import_.converted())
    assert target_import_string == string


@pytest.mark.parametrize(
    "string, result",
    [
        ("from . import util", "\nfrom VIS_CTI_Autofit.VIS_CTI_Tools import util\n"),
        ("from .. import conf", "\nfrom VIS_CTI_Autofit import conf\n"),
        ("from ..tools import util", "\nfrom VIS_CTI_Autofit.VIS_CTI_Tools import util\n")
    ]
)
def test_relative_import(
        package,
        string,
        result
):
    import_ = Import(
        ast.parse(
            string
        ).body[0],
        parent=package["tools"]["namer"]
    )

    target_import_string = unparse(import_.converted())
    assert target_import_string == result


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
        ast.parse(
            "from autofit.non_linear.samples import NestSamples, Sample"
        ).body[0],
        parent=package
    )
    target_string = "\nfrom VIS_CTI_Autofit.VIS_CTI_NonLinear.VIS_CTI_Samples import NestSamples, Sample\n"
    assert unparse(import_.converted()) == target_string
