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


class TestCombination:
    def test_simple(
            self,
            less_than,
            greater_than
    ):
        assert q.Q(
            "a",
            less_than
        ) & q.Q(
            "a",
            greater_than
        ) == q.Q(
            "a",
            q.And(
                less_than,
                greater_than
            )
        )

    def test_second_level(
            self,
            less_than,
            greater_than
    ):
        first = q.Q("a", less_than)
        second = q.Q(
            'a',
            q.Q('b', greater_than)
        )

        combined = q.Q(
            "a",
            q.And(
                less_than,
                q.Q('b', greater_than)
            )
        )

        assert first & second == combined


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
