import pytest

import autofit as af
from autofit.example.model import Gaussian
from autofit.mapper.prior_model import recursion
from autofit.mapper.prior_model.recursion import DynamicRecursionCache


class Wrapper:
    def __init__(self, item):
        self.item = item


class A:
    def __init__(self, b=None):
        self.b = b


class B:
    def __init__(self, a=None):
        self.a = a


@DynamicRecursionCache()
def dict_recurse(item):
    try:
        for key, value in item.__dict__.items():
            setattr(item, key, dict_recurse(value))
    except AttributeError:
        pass
    return Wrapper(item)


def test_basic():
    a = A(B())
    a.b.a = a
    result = dict_recurse(a)
    assert isinstance(result.item.b.item.a, Wrapper)


def test_sub_recursion():
    a = A()
    b = A()
    c = A()

    a.b = b
    b.b = c
    c.b = b

    result = dict_recurse(a)
    assert isinstance(result, Wrapper)


@DynamicRecursionCache()
def count_recurse(item):
    """
    A decorated function whose result cannot contain a promise unless the graph
    it is walking is genuinely cyclic -- the shape of `path_instances_of_class`
    and `AbstractPriorModel.from_instance`, the two decorated functions in the
    library.
    """
    children = []

    for value in getattr(item, "__dict__", {}).values():
        if isinstance(value, (A, B)):
            children.append(count_recurse(value))

    return children


@pytest.fixture(name="replace_promise_calls")
def make_replace_promise_calls(monkeypatch):
    """
    Counts the `replace_promise` traversals the recursion cache performs, so a
    test can assert the traversal is *skipped*, not merely that its result is
    unchanged.
    """
    calls = []
    original = recursion.replace_promise

    def counting_replace_promise(promise, obj, true_value, seen_objects=None):
        calls.append(promise)
        return original(promise, obj, true_value, seen_objects=seen_objects)

    monkeypatch.setattr(recursion, "replace_promise", counting_replace_promise)

    return calls


@pytest.fixture(name="force_traversal")
def make_force_traversal(monkeypatch):
    """
    Restores the unconditional behaviour -- every promise reports itself as
    used, so `replace_promise` runs on every decorated call as it did before
    the `used` flag existed. The regression tests compare against this rather
    than against a stored string, so they keep testing the two implementations
    against each other rather than a snapshot.
    """

    def force_used(self):
        self.used = True

    def _force():
        monkeypatch.setattr(recursion.RecursionPromise, "__init__", force_used)

    return _force


def test_promise_not_used_when_graph_is_flat(replace_promise_calls):
    a = A(B())

    assert count_recurse(a) == [[]]
    assert replace_promise_calls == []


def test_promise_used_when_graph_is_cyclic(replace_promise_calls):
    a = A(B())
    a.b.a = a

    count_recurse(a)

    assert len(replace_promise_calls) > 0


def test_flat_model_info_unchanged_and_traversal_skipped(
    replace_promise_calls, force_traversal
):
    model = af.Model(Gaussian)

    info = model.info

    assert replace_promise_calls == []

    force_traversal()

    assert model.info == info


def test_self_referential_model_info_and_graph_unchanged(
    replace_promise_calls, force_traversal
):
    gaussian = af.Model(Gaussian)
    model = af.Collection(gaussian=gaussian)
    model.self_ref = model

    info = model.info

    # The case the cache exists for: the promise escapes, so the traversal must
    # still run.
    assert len(replace_promise_calls) > 0
    assert model.self_ref is model
    assert model.gaussian is gaussian
    assert model.prior_count == 3

    force_traversal()

    assert model.info == info
    assert model.self_ref is model
    assert model.gaussian is gaussian
