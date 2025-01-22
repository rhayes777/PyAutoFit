import pytest

import autofit as af


class A:
    def __init__(self, a):
        self.a = a


class B:
    def __init__(self, a, b):
        self.a = a
        self.b = b


@pytest.fixture(name="prior")
def make_prior():
    return af.UniformPrior()


@pytest.fixture(name="model_a")
def make_model_a(prior):
    return af.Model(A, a=prior)


@pytest.fixture(name="collection_a")
def make_collection_a(model_a):
    return af.Collection(a=model_a)


def test_cast_multiple(prior):
    model_a_1 = af.Model(A, a=prior)
    model_a_2 = af.Model(A, a=prior)

    collection = af.Collection(a_1=model_a_1, a_2=model_a_2)

    collection = collection.cast({model_a_1: {"b": 1}, model_a_2: {"b": 2}}, B)

    assert collection.a_1.cls is B
    assert collection.a_2.cls is B

    assert collection.a_1.b == 1
    assert collection.a_2.b == 2


def test_cast(collection_a, model_a, prior):
    result = collection_a.cast({model_a: {"b": 2}}, B)
    assert result.a.cls is B
    assert result.a.b == 2
    assert result.a.a is prior


def test_cast_both(collection_a, model_a):
    result = collection_a.cast({model_a: {"a": 1, "b": 2}}, B)

    assert result.a.a == 1
    assert result.a.b == 2


def test_replace_for_path(collection_a):
    collection = collection_a.replacing_for_path(("a", "a"), 3)
    assert collection.a.a == 3
