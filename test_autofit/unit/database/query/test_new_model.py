from autofit.database import query_model as q


class TestCombination:
    def test_simple(
            self,
            less_than,
            greater_than,
            simple_and
    ):
        assert q.Q(
            "a",
            less_than
        ) & q.Q(
            "a",
            greater_than
        ) == simple_and

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

    def test_complicated(
            self,
            less_than,
            greater_than
    ):
        first = q.Q(
            "a",
            q.Q(
                "b",
                q.And(
                    q.Q(
                        "c",
                        less_than
                    ),
                    greater_than
                )
            )
        )

        second = q.Q(
            "a",
            q.Q(
                "b",
                q.Q(
                    "c",
                    greater_than
                )
            )
        )

        combined = q.Q(
            "a",
            q.Q(
                "b",
                q.And(
                    q.And(
                        q.Q(
                            "c",
                            less_than
                        ),
                        greater_than
                    ),
                    greater_than
                )
            )
        )

        assert first & second == combined
