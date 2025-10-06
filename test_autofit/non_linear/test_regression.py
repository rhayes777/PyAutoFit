import pickle

import dill
import pytest

import autofit as af
from autofit import exc


def test_no_priors():
    search = af.DynestyStatic()
    model = af.Collection()
    with pytest.raises(AssertionError):
        search.fit(model, af.Analysis())


@pytest.fixture(name="search")
def make_search():
    return af.DynestyStatic("name")


def test_serialize_optimiser(search):
    search = pickle.loads(pickle.dumps(search))
    assert search.name == "name"


def test_serialize_grid_search(search):
    grid_search = af.SearchGridSearch(search)
    assert grid_search.logger.name == "GridSearch (name)"
    assert "logger" not in grid_search.__getstate__()

    dumped = dill.dumps(grid_search)
    loaded = dill.loads(dumped)
    assert loaded.logger is not None


@pytest.fixture(name="model")
def make_model():
    one = af.Model(af.Gaussian)
    two = af.Model(af.Gaussian)
    model = af.Collection(
        one=one,
        two=two,
    )
    model.add_assertion(one.centre < two.centre)
    return model


def test_skip_assertions(model):
    with pytest.raises(exc.FitException):
        model.instance_from_prior_medians()

    model.instance_from_prior_medians(ignore_prior_limits=True)


def test_recursive_skip_assertions(model):
    model = af.Collection(model=model)
    with pytest.raises(exc.FitException):
        model.instance_from_prior_medians()

    model.instance_from_prior_medians(ignore_prior_limits=True)
