import numpy as np
import pytest

import autofit as af
from autofit import exc


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


def test_as_argument(prior_1, prior_2):
    model = af.Collection(truth=prior_1 < prior_2)

    result = model.instance_for_arguments({prior_1: 0, prior_2: 1})
    assert result.truth is True

    result = model.instance_for_arguments({prior_1: 1, prior_2: 0})
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
        assert (
                lower_assertion.instance_for_arguments({prior_1: 0.3, prior_2: 0.5}) is True
        )
        assert (
                lower_assertion.instance_for_arguments({prior_1: 0.6, prior_2: 0.5})
                is False
        )

    def test_assert_on_arguments_greater(self, greater_assertion, prior_1, prior_2):
        assert (
                greater_assertion.instance_for_arguments({prior_1: 0.6, prior_2: 0.5})
                is True
        )
        assert (
                greater_assertion.instance_for_arguments({prior_1: 0.3, prior_2: 0.5})
                is False
        )

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
        """
        The two halves are combined with `np.logical_and` rather than a Python `and` -- `and`
        coerces its operands to Python bools, which is illegal on a JAX tracer -- so the result
        is an `np.bool_` rather than the `True`/`False` singletons.
        """
        assertion = (0.2 < prior_1) < 0.5
        assert assertion.instance_for_arguments({prior_1: 0.3}) == True
        assert assertion.instance_for_arguments({prior_1: 0.1}) == False
        assert assertion.instance_for_arguments({prior_1: 0.6}) == False


@pytest.fixture(name="promise_model")
def make_promise_model(phase):
    return phase.result.model.one.component


@pytest.fixture(name="model")
def make_model(collection):
    return collection.last.model.one.component


class TestModel:
    def test_assertion_in_model(self, prior_1, prior_2):
        model = af.ModelMapper()
        model.one = prior_1
        model.two = prior_2

        model.add_assertion(prior_1 < prior_2)

        model.instance_from_unit_vector([0.1, 0.2])
        with pytest.raises(af.exc.FitException):
            model.instance_from_unit_vector([0.2, 0.1])

    def test_numerical(self):
        model = af.ModelMapper()
        model.add_assertion(True)
        model.instance_from_unit_vector([])

        model = af.ModelMapper()
        model.add_assertion(False)
        with pytest.raises(exc.FitException):
            model.instance_from_unit_vector([])


class TestAssertionsSatisfiedFromVector:
    """
    The **value** form of the assertion check, which is what makes assertions work under JAX.

    `check_assertions` signals a violation by raising, and a `raise` cannot happen inside a
    trace. `assertions_satisfied_from_vector` returns the same verdict as a boolean array
    instead, which `Fitness` applies with an `xp.where`. These tests pin it on numpy, where the
    verdict must match what `instance_from_vector` does or does not raise; the traced behaviour
    is pinned in `test_autofit/non_linear/test_fitness_assertions_jax.py`.
    """

    @staticmethod
    def _collection(*names):
        return af.Collection(**{name: af.Model(af.ex.Gaussian) for name in names})

    def test_greater_than_assertion(self):
        model = self._collection("gaussian_0", "gaussian_1")
        model.add_assertion(model.gaussian_0.centre > model.gaussian_1.centre)

        # The vector is ordered by prior id: centre, normalization, sigma for each Gaussian in
        # turn, so index 0 is `gaussian_0.centre` and index 3 is `gaussian_1.centre`.
        satisfied = model.assertions_satisfied_from_vector(
            [20.0, 1.0, 1.0, 10.0, 1.0, 1.0]
        )
        violated = model.assertions_satisfied_from_vector(
            [10.0, 1.0, 1.0, 20.0, 1.0, 1.0]
        )

        assert np.asarray(satisfied).dtype == bool
        assert bool(satisfied) is True
        assert bool(violated) is False

    def test_greater_than_equal_assertion(self):
        """
        The `>=` assertion is a different class to `>`, and the equal case is the one that
        distinguishes them.
        """
        model = self._collection("gaussian_0", "gaussian_1")
        model.add_assertion(model.gaussian_0.centre >= model.gaussian_1.centre)

        equal = model.assertions_satisfied_from_vector(
            [10.0, 1.0, 1.0, 10.0, 1.0, 1.0]
        )

        assert bool(equal) is True
        assert (
            bool(
                model.assertions_satisfied_from_vector(
                    [20.0, 1.0, 1.0, 10.0, 1.0, 1.0]
                )
            )
            is True
        )
        assert (
            bool(
                model.assertions_satisfied_from_vector(
                    [10.0, 1.0, 1.0, 20.0, 1.0, 1.0]
                )
            )
            is False
        )

    def test_compound_assertion(self):
        """
        `(a > b) > c` chains to `a > b and b > c`, a `CompoundAssertion`. Both halves must hold,
        and the combination must not go through a Python `and`.
        """
        model = self._collection("gaussian_0", "gaussian_1", "gaussian_2")
        model.add_assertion(
            (model.gaussian_0.centre > model.gaussian_1.centre)
            > model.gaussian_2.centre
        )

        ordered = [3.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        second_half_violated = [3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0]
        first_half_violated = [1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.5, 1.0, 1.0]

        assert bool(model.assertions_satisfied_from_vector(ordered)) is True
        assert (
            bool(model.assertions_satisfied_from_vector(second_half_violated)) is False
        )
        assert (
            bool(model.assertions_satisfied_from_vector(first_half_violated)) is False
        )

    def test_compound_prior_assertion(self):
        """
        The motivating case: an ordering between two *derived* quantities rather than two bare
        priors -- here the squared radius of each component's (centre, normalization) pair, which
        is the 1D stand-in for the MGE `ell_comps` label degeneracy the assertion was added to
        break. The assertion adds no free parameters, so the vector length is unchanged.
        """
        model = self._collection("gaussian_0", "gaussian_1")

        prior_count = model.prior_count
        model.add_assertion(
            (model.gaussian_0.centre**2 + model.gaussian_0.normalization**2)
            > (model.gaussian_1.centre**2 + model.gaussian_1.normalization**2)
        )

        assert model.prior_count == prior_count

        # |(0.4, 0.3)|^2 = 0.25 > |(0.05, 0.05)|^2 = 0.005
        satisfied = [0.4, 0.3, 1.0, 0.05, 0.05, 1.0]
        violated = [0.05, 0.05, 1.0, 0.4, 0.3, 1.0]

        assert bool(model.assertions_satisfied_from_vector(satisfied)) is True
        assert bool(model.assertions_satisfied_from_vector(violated)) is False

    def test_model_without_assertions_is_always_satisfied(self):
        """
        A model with nothing attached must return `True` rather than an empty reduction, so
        `Fitness` can call this unconditionally.
        """
        model = self._collection("gaussian_0")

        satisfied = model.assertions_satisfied_from_vector([1.0, 1.0, 1.0])

        assert np.asarray(satisfied).dtype == bool
        assert bool(satisfied) is True

    def test_vector_length_is_checked(self):
        """
        The same guard `instance_from_vector` applies, since the two must map a vector to
        arguments identically or the assertion would be evaluated against the wrong parameters.
        """
        model = self._collection("gaussian_0")
        model.add_assertion(model.gaussian_0.centre > 0.0)

        with pytest.raises(AssertionError):
            model.assertions_satisfied_from_vector([1.0, 1.0])

    def test_agrees_with_check_assertions(self):
        """
        The value form and the exception form are two statements of one rule, so a vector that
        `instance_from_vector` rejects must be exactly the vector this returns `False` for.
        """
        model = self._collection("gaussian_0", "gaussian_1")
        model.add_assertion(model.gaussian_0.centre > model.gaussian_1.centre)

        satisfied = [20.0, 1.0, 1.0, 10.0, 1.0, 1.0]
        violated = [10.0, 1.0, 1.0, 20.0, 1.0, 1.0]

        model.instance_from_vector(satisfied)
        assert bool(model.assertions_satisfied_from_vector(satisfied)) is True

        with pytest.raises(exc.FitException):
            model.instance_from_vector(violated)
        assert bool(model.assertions_satisfied_from_vector(violated)) is False


class TestGatheredAssertions:
    """
    Assertions are attached to whichever model object the user held, which is very often a child
    component rather than the model the search is handed. `check_assertions` never had to care --
    every model checks its own as its instance is built -- but the traced path is handed one model
    and has to find them itself, so a walk that missed a child would enforce fewer assertions
    under JAX than under numpy. That asymmetry is what these pin against.
    """

    def test_own_assertions(self):
        model = af.Collection(gaussian=af.Model(af.ex.Gaussian))
        assertion = model.gaussian.centre > 0.0
        model.add_assertion(assertion)

        assert model.gathered_assertions() == [assertion]

    def test_child_assertions(self):
        """
        The form the pre-existing numpy regression test uses: attached to the child `Model`, and
        the `Collection` is what the search sees.
        """
        gaussian_0 = af.Model(af.ex.Gaussian)
        gaussian_1 = af.Model(af.ex.Gaussian)
        assertion = gaussian_0.centre > gaussian_1.centre
        gaussian_0.add_assertion(assertion)

        model = af.Collection(gaussian_0=gaussian_0, gaussian_1=gaussian_1)

        assert model._assertions == []
        assert model.gathered_assertions() == [assertion]

    def test_nested_collections_and_own_assertions_first(self):
        gaussian_0 = af.Model(af.ex.Gaussian)
        gaussian_1 = af.Model(af.ex.Gaussian)
        child_assertion = gaussian_0.centre > gaussian_1.centre
        gaussian_0.add_assertion(child_assertion)

        inner = af.Collection(gaussian_0=gaussian_0, gaussian_1=gaussian_1)
        inner_assertion = inner.gaussian_1.sigma > 0.5
        inner.add_assertion(inner_assertion)

        model = af.Collection(inner=inner)
        own_assertion = model.inner.gaussian_0.normalization > 0.0
        model.add_assertion(own_assertion)

        gathered = model.gathered_assertions()

        assert gathered[0] is own_assertion
        assert set(map(id, gathered)) == {
            id(own_assertion),
            id(inner_assertion),
            id(child_assertion),
        }

    def test_model_without_assertions(self):
        model = af.Collection(gaussian=af.Model(af.ex.Gaussian))

        assert model.gathered_assertions() == []

    def test_a_shared_component_contributes_once(self):
        """
        The same `Model` reachable by two paths is one component with one set of assertions, so
        its assertion must not be evaluated (or counted) twice.
        """
        gaussian = af.Model(af.ex.Gaussian)
        assertion = gaussian.centre > 0.0
        gaussian.add_assertion(assertion)

        model = af.Collection(first=gaussian, second=gaussian)

        assert model.gathered_assertions() == [assertion]

    def test_child_assertions_are_evaluated_from_a_vector(self):
        """
        The verdict, not just the gathering: a child-attached assertion must make
        `assertions_satisfied_from_vector` on the *parent* return `False`, matching the
        `FitException` `instance_from_vector` raises for the same vector.
        """
        gaussian_0 = af.Model(af.ex.Gaussian)
        gaussian_1 = af.Model(af.ex.Gaussian)
        gaussian_0.add_assertion(gaussian_0.centre > gaussian_1.centre)
        model = af.Collection(gaussian_0=gaussian_0, gaussian_1=gaussian_1)

        satisfied = [20.0, 1.0, 1.0, 10.0, 1.0, 1.0]
        violated = [10.0, 1.0, 1.0, 20.0, 1.0, 1.0]

        assert bool(model.assertions_satisfied_from_vector(satisfied)) is True
        assert bool(model.assertions_satisfied_from_vector(violated)) is False

        with pytest.raises(exc.FitException):
            model.instance_from_vector(violated)

    def test_explicit_assertions_argument_is_used_verbatim(self):
        """
        `Fitness` gathers once in its constructor and passes the list back in, so the per-call
        cost is the evaluation rather than a fresh walk of the model tree.
        """
        gaussian_0 = af.Model(af.ex.Gaussian)
        gaussian_1 = af.Model(af.ex.Gaussian)
        gaussian_0.add_assertion(gaussian_0.centre > gaussian_1.centre)
        model = af.Collection(gaussian_0=gaussian_0, gaussian_1=gaussian_1)

        violated = [10.0, 1.0, 1.0, 20.0, 1.0, 1.0]

        assert (
            bool(
                model.assertions_satisfied_from_vector(
                    violated, assertions=model.gathered_assertions()
                )
            )
            is False
        )
        # An empty list is not "no list": it means evaluate nothing.
        assert (
            bool(model.assertions_satisfied_from_vector(violated, assertions=[]))
            is True
        )
