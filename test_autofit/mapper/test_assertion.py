import pytest

import autofit as af
from autofit.mapper.prior_model import assertion as a


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


class TestAssertion:
    def test_lower_equal_assertion(self, prior_1, prior_2):
        assertion = prior_1 <= prior_2
        assertion({prior_1: 0.4, prior_2: 0.5})
        assertion({prior_1: 0.5, prior_2: 0.5})
        with pytest.raises(af.exc.FitException):
            assertion({prior_1: 0.6, prior_2: 0.5})

    def test_greater_equal_assertion(self, prior_1, prior_2):
        assertion = prior_1 >= prior_2
        assertion({prior_1: 0.6, prior_2: 0.5})
        assertion({prior_1: 0.5, prior_2: 0.5})
        with pytest.raises(af.exc.FitException):
            assertion({prior_1: 0.4, prior_2: 0.5})

    def test_lower_assertion(self, lower_assertion, prior_1, prior_2):
        assert isinstance(lower_assertion, a.GreaterThanLessThanAssertion)

        assert lower_assertion._lower is prior_1
        assert lower_assertion._greater is prior_2

    def test_greater_assertion(self, greater_assertion, prior_1, prior_2):
        assert isinstance(greater_assertion, a.GreaterThanLessThanAssertion)

        assert greater_assertion._lower is prior_2
        assert greater_assertion._greater is prior_1

    def test_assert_on_arguments_lower(self, lower_assertion, prior_1, prior_2):
        lower_assertion({prior_1: 0.3, prior_2: 0.5})
        with pytest.raises(af.exc.FitException):
            lower_assertion({prior_1: 0.6, prior_2: 0.5})

    def test_assert_on_arguments_greater(self, greater_assertion, prior_1, prior_2):
        greater_assertion({prior_1: 0.6, prior_2: 0.5})
        with pytest.raises(af.exc.FitException):
            greater_assertion({prior_1: 0.3, prior_2: 0.5})

    def test_numerical_assertion(self, prior_1):
        assertion = prior_1 < 0.5

        assertion({prior_1: 0.4})
        with pytest.raises(af.exc.FitException):
            assertion({prior_1: 0.6})

    def test_numerical_assertion_left(self, prior_1):
        assertion = 0.5 < prior_1

        assertion({prior_1: 0.6})
        with pytest.raises(af.exc.FitException):
            assertion({prior_1: 0.4})
        with pytest.raises(af.exc.FitException):
            assertion({prior_1: 0.5})

    def test_compound_assertion(self, prior_1):
        assertion = (0.2 < prior_1) < 0.5
        assertion({prior_1: 0.3})
        with pytest.raises(af.exc.FitException):
            assertion({prior_1: 0.1})
        with pytest.raises(af.exc.FitException):
            assertion({prior_1: 0.6})


@pytest.fixture(name="promise_model")
def make_promise_model(phase):
    return phase.result.model.one.light


@pytest.fixture(name="model")
def make_model(collection):
    return collection.last.model.one.light


class TestPromiseAssertion:
    def test_less_than(self, promise_model, collection, model):
        promise = promise_model.axis_ratio < promise_model.phi
        assert isinstance(promise, af.AssertionPromise)

        assertion = promise.populate(collection)
        assert isinstance(assertion, af.GreaterThanLessThanAssertion)
        assert model.axis_ratio == assertion._lower
        assert model.phi == assertion._greater

    def test_greater_than(self, promise_model, collection, model):
        promise = promise_model.axis_ratio > promise_model.phi
        # noinspection PyUnresolvedReferences
        assertion = promise.populate(collection)
        assert model.axis_ratio == assertion._greater
        assert model.phi == assertion._lower


def test_assertion_in_model(prior_1, prior_2):
    model = af.ModelMapper()
    model.one = prior_1
    model.two = prior_2

    model.add_assertion(prior_1 < prior_2)

    model.instance_from_unit_vector([0.1, 0.2])
    with pytest.raises(af.exc.FitException):
        model.instance_from_unit_vector([0.2, 0.1])
