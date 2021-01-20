from autofit.database import query_model as q


class TestCombination:
    def test_simple(self):
        less_than = q.V(
            "<", 1
        )
        greater_than = q.V(
            ">", 0
        )

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
