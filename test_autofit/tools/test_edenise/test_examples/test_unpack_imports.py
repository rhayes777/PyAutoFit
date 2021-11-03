import pytest

from autofit.tools.edenise import File, Import, LineItem


@pytest.fixture(
    name="aliased_import"
)
def make_aliased_import():
    return Import.parse_fragment(
        "import autofit as af"
    )


def test_alias_import(
        aliased_import
):
    assert aliased_import.is_aliased


def test_alias(
        aliased_import
):
    assert aliased_import.alias == "af"


def test_not_alias_import():
    assert Import.parse_fragment(
        "import autofit"
    ).is_aliased is False


@pytest.fixture(
    name="file"
)
def make_file(
        package,
        examples_directory,
        eden_output_directory
):
    return File(
        examples_directory / "unpack_imports.py",
        parent=package,
        prefix=""
    )


def test_alias_imports(
        file
):
    alias_import, = file.aliased_imports
    assert alias_import.alias == "af"

    assert file.aliases == ["af"]


def test_attributes(
        file
):
    assert len(list(file.attributes())) == 3


def test_attributes_for_alias(
        file
):
    assert file.attributes_for_alias(
        "af"
    ) == {
        "Model",
        "Gaussian"
    }


def test_whole_file(
        file
):
    assert file.target_string == """
from VIS_CTI_Autofit import Model, Gaussian
model = Model(Gaussian)
print(model.prior_count)
"""


def test_replace_alias(
        file
):
    item = LineItem.parse_fragment(
        """model = af.Model(
    af.Gaussian
)
        """,
        parent=file
    )

    assert item.target_string == """
model = Model(Gaussian)
"""

