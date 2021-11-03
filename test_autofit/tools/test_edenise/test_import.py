import ast

import pytest
from astunparse import unparse

from autofit.tools.edenise import Import


@pytest.fixture(
    name="import_"
)
def make_import(file):
    return Import(
        ast.parse(
            "from autofit.tools.edenise import Line"
        ).body[0],
        parent=file
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
        file,
        source,
        target
):
    import_ = Import(
        ast.parse(
            source
        ).body[0],
        parent=file
    )
    target_string = target
    assert unparse(import_.converted()) == target_string


def test_project_import(import_, file):
    import_.parent = file["tools"]["edenise"]["converter"]
    assert import_.is_in_project is True


def test_non_project_import(file):
    import_ = Import(
        ast.parse("import os").body[0],
        parent=file
    )
    import_.parent = file
    assert import_.is_in_project is False
    assert import_.target_string == """
import os
"""


def test_non_project_from_import(
        file
):
    import_ = Import(
        ast.parse("from hashlib import md5").body[0],
        parent=file
    )
    assert import_.target_string == """
from hashlib import md5
"""


def test_target_import_string(import_):
    string = "\nfrom VIS_CTI_Autofit.VIS_CTI_Tools.VIS_CTI_Edenise import Line\n"
    target_import_string = unparse(import_.converted())
    assert target_import_string == string


def test_multi_import(
        file
):
    import_ = Import(
        ast.parse(
            "from autofit.tools.edenise import file, File, Import"
        ).body[0],
        parent=file
    )
    string = "\nfrom VIS_CTI_Autofit.VIS_CTI_Tools.VIS_CTI_Edenise import file, File, Import\n"
    target_import_string = unparse(import_.converted())
    assert target_import_string == string


def test_import_as(
        file
):
    import_ = Import(
        ast.parse(
            "from autofit.tools import edenise as e"
        ).body[0],
        parent=file
    )
    string = "\nfrom VIS_CTI_Autofit.VIS_CTI_Tools import VIS_CTI_Edenise as e\n"
    target_import_string = unparse(import_.converted())
    assert target_import_string == string


def test_file_import(
        file
):
    import_ = Import(
        ast.parse(
            "from autofit.tools import edenise"
        ).body[0],
        parent=file
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
        file
):
    file._should_rename_modules = False
    import_ = Import(
        ast.parse(
            "from autofit.non_linear.samples import NestSamples, Sample"
        ).body[0],
        parent=file
    )
    target_string = "\nfrom VIS_CTI_Autofit.VIS_CTI_NonLinear.VIS_CTI_Samples import NestSamples, Sample\n"
    assert unparse(import_.converted()) == target_string
