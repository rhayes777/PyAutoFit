from shutil import rmtree

import pytest


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
