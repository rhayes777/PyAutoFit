import pytest

from autofit.tools.edenise import File, Import


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


def test_unpack_imports(
        package,
        examples_directory,
        eden_output_directory
):
    file = File(
        examples_directory / "unpack_imports.py",
        parent=package,
        prefix=""
    )

    alias_import, = file.alias_imports
    assert alias_import.alias == "af"
