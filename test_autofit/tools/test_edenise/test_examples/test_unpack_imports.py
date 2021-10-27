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


def test_uses_alias(
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

