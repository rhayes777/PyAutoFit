from autofit.tools.edenise import LineItem
import pytest


@pytest.mark.parametrize(
    "string, boolean",
    [
        ("import autofit", True),
        ("from .. import something", True),
        ("from autofit import something", True),
        ("from other import something", False),
        ("from .other import something", True),
        ("from autoconf import something", True),
        ("import autoconf", True),
    ]
)
def test_is_in_project(
        string,
        boolean,
        package
):
    assert LineItem.parse_fragment(
        string,
        parent=package
    ).is_in_project == boolean
