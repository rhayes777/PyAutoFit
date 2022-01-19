from pathlib import Path

import pytest

from autofit.tools.edenise import Package

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
        is_top_level=True,
        eden_dependencies=["autoconf"],
        should_rename_modules=True
    )


@pytest.fixture(
    name="file"
)
def make_file(
        package
):
    return package[
        "__init__"
    ]


directory = Path(
    __file__
).parent


@pytest.fixture(
    name="examples_directory"
)
def make_examples_directory():
    return directory / "examples"


@pytest.fixture(
    name="eden_output_directory"
)
def make_eden_output_directory():
    return directory / "output"
