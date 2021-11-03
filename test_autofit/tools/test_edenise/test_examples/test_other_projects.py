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


@pytest.mark.parametrize(
    "source, target",
    [
        (
                "from autoconf.class_path import get_class",
                "from VIS_CTI_Autoconf.class_path import get_class"
        ),
        (
                "from autoconf.tools import decorators",
                "from VIS_CTI_Autoconf.VIS_CTI_Tools import decorators"
        ),
        (
                "import autoconf.tools.decorators",
                "import VIS_CTI_Autoconf.VIS_CTI_Tools.decorators"
        ),
        (
                "import autoconf.tools",
                "import VIS_CTI_Autoconf.VIS_CTI_Tools"
        )
    ]
)
def test_import(
        file,
        source,
        target
):
    import_ = Import.parse_fragment(
        source,
        parent=file
    )
    assert import_.target_string.strip(" \n") == target.strip(" \n")
