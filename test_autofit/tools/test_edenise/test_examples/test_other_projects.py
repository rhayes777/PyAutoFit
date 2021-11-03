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
        examples_directory / "other_projects.py",
        parent=package,
        prefix=""
    )


def test_other_projects(
        file
):
    assert file.target_string == """
from VIS_CTI_Autoconf.class_path import get_class
from VIS_CTI_Autofit.VIS_CTI_Mapper import Prior
get_class(Prior)
"""


def test_import(
        file
):
    import_ = Import.parse_fragment(
        "from autoconf.class_path import get_class",
        parent=file
    )
    assert import_.target_string == """
from VIS_CTI_Autoconf.class_path import get_class
"""


def test_package(
        file
):
    import_ = Import.parse_fragment(
        "from autoconf.tools import decorators",
        parent=file
    )
    assert import_.target_string == """
from VIS_CTI_Autoconf.VIS_CTI_Tools import decorators
"""
