import pytest

import autofit as af
from test_autofit import mock

names = ["one", "two", "three"]


@pytest.fixture(
    name="collection"
)
def make_collection():
    return af.Collection({
        name: af.PriorModel(
            mock.MockClassx2
        )
        for name in names
    })


def test_prior_count(collection):
    assert collection.prior_count == 6


@pytest.mark.parametrize(
    "name",
    names
)
def test_children(collection, name):
    assert getattr(collection, name).prior_count == 2


def test_replace(collection):
    collection.one = af.PriorModel(
        mock.MockClassx4
    )

    assert collection.one.prior_count == 4
    assert collection.prior_count == 8
