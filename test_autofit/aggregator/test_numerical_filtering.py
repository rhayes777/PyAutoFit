import pytest


@pytest.fixture(name="ages_for_predicate")
def make_ages_for_predicate(path_aggregator):
    def ages_for_predicate(predicate):
        result = path_aggregator.filter(predicate)
        return [child.age for child in result.values("child")]

    return ages_for_predicate


class TestNumericalFiltering:
    def test_equality(self, path_aggregator):
        predicate = path_aggregator.child.age == 17
        result = path_aggregator.filter(predicate)
        assert len(result) == 1
        assert list(result.values("child"))[0].age == 17

    def test_greater_than(self, ages_for_predicate, path_aggregator):
        ages = ages_for_predicate(path_aggregator.child.age > 10)
        assert ages == [17]

    def test_greater_than_equal(self, ages_for_predicate, path_aggregator):
        ages = ages_for_predicate(path_aggregator.child.age >= 10)
        assert set(ages) == {10, 17}

    def test_less_than(self, ages_for_predicate, path_aggregator):
        ages = ages_for_predicate(path_aggregator.child.age < 11)
        assert ages == [10]

    def test_greater_than_rhs(self, ages_for_predicate, path_aggregator):
        ages = ages_for_predicate(10 < path_aggregator.child.age)
        assert ages == [17]

    def test_less_than_rhs(self, ages_for_predicate, path_aggregator):
        ages = ages_for_predicate(11 > path_aggregator.child.age)
        assert ages == [10]

    def test_aggregator_to_aggregator(self, path_aggregator):
        predicate = path_aggregator.child.age == path_aggregator.child.age
        assert len(path_aggregator.filter(predicate)) == 2

        predicate = path_aggregator.child.age > path_aggregator.child.age
        assert len(path_aggregator.filter(predicate)) == 0
