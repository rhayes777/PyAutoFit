import pytest

import autofit as af
from autofit import exc
from test_autofit import mock


@pytest.fixture(name="prior_1")
def make_prior_1():
    return af.UniformPrior()


@pytest.fixture(name="prior_2")
def make_prior_2():
    return af.UniformPrior()


@pytest.fixture(name="lower_assertion")
def make_lower_assertion(prior_1, prior_2):
    return prior_1 < prior_2


@pytest.fixture(name="greater_assertion")
def make_greater_assertion(prior_1, prior_2):
    return prior_1 > prior_2


def test_as_argument(
        prior_1,
        prior_2
):
    model = af.Collection(
        truth=prior_1 < prior_2
    )

    result = model.instance_for_arguments(
        {
            prior_1: 0,
            prior_2: 1
        }
    )
    assert result.truth is True

    result = model.instance_for_arguments(
        {
            prior_1: 1,
            prior_2: 0
        }
    )
    assert result.truth is False


class TestAssertion:
    def test_lower_equal_assertion(self, prior_1, prior_2):
        assertion = prior_1 <= prior_2
        assert assertion.instance_for_arguments({prior_1: 0.4, prior_2: 0.5}) is True
        assert assertion.instance_for_arguments({prior_1: 0.5, prior_2: 0.5}) is True
        assert assertion.instance_for_arguments({prior_1: 0.6, prior_2: 0.5}) is False

    def test_greater_equal_assertion(self, prior_1, prior_2):
        assertion = prior_1 >= prior_2
        assert assertion.instance_for_arguments({prior_1: 0.6, prior_2: 0.5}) is True
        assert assertion.instance_for_arguments({prior_1: 0.5, prior_2: 0.5}) is True

        assert assertion.instance_for_arguments({prior_1: 0.4, prior_2: 0.5}) is False

    def test_assert_on_arguments_lower(self, lower_assertion, prior_1, prior_2):
        assert lower_assertion.instance_for_arguments({prior_1: 0.3, prior_2: 0.5}) is True
        assert lower_assertion.instance_for_arguments({prior_1: 0.6, prior_2: 0.5}) is False

    def test_assert_on_arguments_greater(self, greater_assertion, prior_1, prior_2):
        assert greater_assertion.instance_for_arguments({prior_1: 0.6, prior_2: 0.5}) is True
        assert greater_assertion.instance_for_arguments({prior_1: 0.3, prior_2: 0.5}) is False

    def test_numerical_assertion(self, prior_1):
        assertion = prior_1 < 0.5

        assert assertion.instance_for_arguments({prior_1: 0.4}) is True
        assert assertion.instance_for_arguments({prior_1: 0.6}) is False

    def test_numerical_assertion_left(self, prior_1):
        assertion = 0.5 < prior_1

        assert assertion.instance_for_arguments({prior_1: 0.6}) is True
        assert assertion.instance_for_arguments({prior_1: 0.4}) is False
        assert assertion.instance_for_arguments({prior_1: 0.5}) is False

    def test_compound_assertion(self, prior_1):
        assertion = (0.2 < prior_1) < 0.5
        assert assertion.instance_for_arguments({prior_1: 0.3}) is True
        assert assertion.instance_for_arguments({prior_1: 0.1}) is False
        assert assertion.instance_for_arguments({prior_1: 0.6}) is False

    # noinspection PyUnresolvedReferences
    def test_annotation(self):
        model = af.PriorModel(mock.MockDistanceClass)
        assertion = model.one < model.two

        assert assertion.instance_for_arguments({
            model.one.value: 0.3,
            model.two.value: 0.4
        }) is True
        assert assertion.instance_for_arguments({
            model.one.value: 0.5,
            model.two.value: 0.4
        }) is False


@pytest.fixture(name="promise_model")
def make_promise_model(phase):
    return phase.result.model.one.component


@pytest.fixture(name="model")
def make_model(collection):
    return collection.last.model.one.component


class TestPromiseAssertion:
    def test_less_than(self, promise_model, collection, model):
        promise = promise_model.one < promise_model.two
        assert isinstance(promise, af.prior.GreaterThanLessThanAssertion)

        assertion = promise.populate(collection)
        assert isinstance(assertion, af.prior.GreaterThanLessThanAssertion)

    def test_greater_than(self, promise_model, collection, model):
        promise = promise_model.one > promise_model.two
        assert isinstance(promise, af.prior.GreaterThanLessThanAssertion)

    def test_greater_than_equal(self, promise_model, collection, model):
        promise = promise_model.one >= promise_model.two
        assert isinstance(promise, af.prior.GreaterThanLessThanEqualAssertion)

    def test_integer_promise_assertion(self, promise_model, collection, model):
        promise = promise_model.one > 1.0
        assert isinstance(promise, af.prior.GreaterThanLessThanAssertion)

    def test_compound_assertion(self, promise_model, collection, model):
        promise = (1.0 < promise_model.one) < 1.0
        assert isinstance(promise, af.prior.CompoundAssertion)

        assertion = promise.populate(collection)
        assert isinstance(assertion, af.prior.CompoundAssertion)


class TestModel:
    def test_assertion_in_model(self, prior_1, prior_2):
        model = af.ModelMapper()
        model.one = prior_1
        model.two = prior_2

        model.add_assertion(prior_1 < prior_2)

        model.instance_from_unit_vector([0.1, 0.2])
        with pytest.raises(af.exc.FitException):
            model.instance_from_unit_vector([0.2, 0.1])

    def test_populate_assertion_promises(self, promise_model, collection):
        model = af.ModelMapper()
        model.component = promise_model
        # noinspection PyTypeChecker
        model.add_assertion(promise_model.one < promise_model.two)

        model = model.populate(collection)

        assert model._assertions[0].instance_for_arguments(
            {
                model.component.one: 0.0,
                model.component.two: 1.0
            }
        ) is True
        assert model._assertions[0].instance_for_arguments(
            {
                model.component.one: 1.0,
                model.component.two: 0.0
            }
        ) is False

    def test_numerical(self):
        model = af.ModelMapper()
        model.add_assertion(True)
        model.instance_from_unit_vector([])

        model = af.ModelMapper()
        model.add_assertion(False)
        with pytest.raises(exc.FitException):
            model.instance_from_unit_vector([])
