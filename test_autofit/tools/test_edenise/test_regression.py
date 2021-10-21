from pathlib import Path
from shutil import rmtree

import pytest

from autofit.tools.edenise import File

directory = Path(
    __file__
).parent
examples_directory = directory / "examples"
output_directory = directory / "output"


@pytest.fixture(
    autouse=True
)
def make_directories_and_clean(
        package
):
    package._generate_directory(
        output_directory
    )
    yield
    rmtree(
        output_directory
    )


def test_new_line_bracket(
        package
):
    module_path = examples_directory / "new_line_bracket.py"
    file = File(
        module_path,
        prefix="",
        parent=package
    )

    package._generate_directory(
        output_directory
    )
    file.generate_target(
        output_directory
    )
    assert output_directory / package.target_file_name / "VIS_CTI_NewLineBracket.py"
