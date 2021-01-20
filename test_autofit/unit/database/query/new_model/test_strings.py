from autofit.database import query_model as q


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
