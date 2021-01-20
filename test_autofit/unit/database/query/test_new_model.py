import pytest

from autofit.database import query_model as q


@pytest.fixture(
    name="less_than"
)
def make_less_than():
    return q.V(
        "<", 1
    )


@pytest.fixture(
    name="greater_than"
)
def make_greater_than():
    return q.V(
        ">", 0
    )


@pytest.fixture(
    name="simple_combination"
)
def make_simple_combination(
        less_than,
        greater_than
):
    return q.Q(
        "a",
        q.And(
            less_than,
            greater_than
        )
    )


@pytest.fixture(
    name="second_level"
)
def make_second_level(
        less_than,
        greater_than
):
    return q.Q(
        "a",
        q.And(
            less_than,
            q.Q('b', greater_than)
        )
    )


class TestCombination:
    def test_simple(
            self,
            less_than,
            greater_than,
            simple_combination
    ):
        assert q.Q(
            "a",
            less_than
        ) & q.Q(
            "a",
            greater_than
        ) == simple_combination

    def test_second_level(
            self,
            less_than,
            greater_than,
            second_level
    ):
        first = q.Q("a", less_than)
        second = q.Q(
            'a',
            q.Q('b', greater_than)
        )

        assert first & second == second_level


class TestString:
    def test_named_query(self):
        query = q.Q(
            "a"
        )

        assert query == (
            "SELECT parent_id "
            "FROM object AS o "
            "WHERE o.name = 'a'"
        )

    def test_with_value(self):
        query = q.Q(
            "a",
            q.V(
                "=",
                1
            )
        )

        assert query == (
            "SELECT parent_id "
            "FROM object AS o, "
            "value AS v "
            "WHERE o.name = 'a' "
            "AND v.value = 1"
        )

    def test_simple_combination(
            self,
            simple_combination
    ):
        assert simple_combination == (
            "SELECT parent_id "
            "FROM object AS o, "
            "value AS v "
            "WHERE o.name = 'a' "
            "AND v.value < 1 "
            "AND v.value > 0"
        )

    def test_second_level(
            self,
            second_level
    ):
        assert second_level == (
            "SELECT parent_id "
            "FROM object AS o, "
            "value AS v "
            "WHERE o.name = 'a' "
            "AND v.value < 1 "
            "AND id IN ("
            "SELECT parent_id "
            "FROM object AS o, "
            "value AS v "
            "WHERE o.name = 'b' "
            "AND v.value > 0"
            ")"
        )
