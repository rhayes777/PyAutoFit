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
