from shutil import rmtree

import pytest

from autofit.tools.edenise import File


@pytest.fixture(
    autouse=True
)
def make_directories_and_clean(
        package,
        eden_output_directory
):
    package._generate_directory(
        eden_output_directory
    )
    yield
    rmtree(
        eden_output_directory
    )


def test_new_line_bracket(
        package,
        examples_directory,
        eden_output_directory
):
    module_path = examples_directory / "new_line_bracket.py"
    file = File(
        module_path,
        prefix="",
        parent=package
    )

    package._generate_directory(
        eden_output_directory
    )
    file.generate_target(
        eden_output_directory
    )
    assert eden_output_directory / package.target_file_name / "VIS_CTI_NewLineBracket.py"
