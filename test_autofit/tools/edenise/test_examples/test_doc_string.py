import pytest

from autofit.tools.edenise import File


@pytest.fixture(
    name="file"
)
def make_file(
        package,
        examples_directory
):
    return File(
        examples_directory / "doc_string.py",
        parent=package,
        prefix=""
    )


def test_doc_string(file):
    assert file.target_string == """
\"\"\"
This
is
a
doc's
string
\"\"\"
"""
