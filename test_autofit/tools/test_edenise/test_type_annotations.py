import pytest

from autofit.tools.edenise.structure import LineItem


@pytest.fixture(
    autouse=True
)
def set_flag(package):
    package._should_remove_type_annotations = True


def test_dotted_annotation(
        package
):
    assert LineItem.parse_fragment(
        "def my_func() -> np.ndarray:\n    pass",
        parent=package
    ).target_string == """

def my_func():
    pass
"""


def test_ellipsis(
        package
):
    assert LineItem.parse_fragment(
        "def my_func(t: Tuple[int, ...]) -> Tuple[int, ...]:\n    pass",
        parent=package
    ).target_string == """

def my_func(t):
    pass
"""


def test_lost_argument(
        package
):
    assert LineItem.parse_fragment(
        "def convert_grid_2d(grid_2d: Union[np.ndarray, List], mask_2d):\n    pass",
        parent=package
    ).target_string == """

def convert_grid_2d(grid_2d, mask_2d):
    pass
"""


def test_regression(
        package
):
    assert LineItem.parse_fragment(
        """def path_instances_of_class(
        obj, cls: type, ignore_class: Optional[Union[type, Tuple[type]]] = None
):\n    pass""",
        parent=package
    ).target_string == """

def path_instances_of_class(obj, cls, ignore_class=None):
    pass
"""


class TestStripAnnotations:
    @pytest.mark.parametrize(
        "string",
        [
            "def my_func() -> dict:\n    pass",
            "def my_func()->dict:\n    pass",
            "def my_func() -> dict :\n    pass",
        ]
    )
    def test_strip_return_type(
            self,
            package,
            string
    ):
        assert LineItem.parse_fragment(
            string,
            parent=package
        ).target_string == """

def my_func():
    pass
"""

    def test_across_new_lines(
            self,
            package
    ):
        line_item = LineItem.parse_fragment(
            """def my_func(
                one: dict,
                two: dict
            ):
                pass
            """,
            parent=package
        )
        assert line_item.target_string == """

def my_func(one, two):
    pass
"""

    @pytest.mark.parametrize(
        "annotation",
        [
            "Optional[Union[list, str]]",
            "Optional[Union[type, Tuple[type]]]"
        ]
    )
    def test_complex_type_annotation(
            self,
            package,
            annotation
    ):
        assert LineItem.parse_fragment(
            f"def my_func(complex: {annotation}):\n    pass",
            parent=package
        ).target_string == """

def my_func(complex):
    pass
"""

    def test_dont_convert_dict(
            self,
            package
    ):
        string = "{'one': 1, 'two': 2}"
        assert LineItem.parse_fragment(
            string,
            parent=package
        ).target_string == """
{'one': 1, 'two': 2}
"""

    @pytest.mark.parametrize(
        "string",
        [
            "def my_func() -> dict:\n    pass",
            "def my_func()->dict:\n    pass",
            "def my_func() -> dict :\n    pass",
        ]
    )
    def test_strip_return_type(
            self,
            package,
            string
    ):
        assert LineItem.parse_fragment(
            string,
            parent=package
        ).target_string == """

def my_func():
    pass
"""

    def test_multiple_arguments(
            self,
            package
    ):
        assert LineItem.parse_fragment(
            "def my_func(arg1: dict, arg2: dict):\n    pass",
            parent=package
        ).target_string == """

def my_func(arg1, arg2):
    pass
"""

    def test_strip_argument_type(
            self,
            package
    ):
        assert LineItem.parse_fragment(
            "def my_func(arg: dict):\n    pass",
            parent=package
        ).target_string == """

def my_func(arg):
    pass
"""
