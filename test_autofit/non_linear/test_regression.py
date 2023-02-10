import pickle

import dill
import pytest

import autofit as af


def test_no_priors():
    search = af.DynestyStatic()
    model = af.Collection()
    with pytest.raises(AssertionError):
        search.fit(model, af.Analysis())


@pytest.fixture(
    name="optimizer"
)
def make_optimizer():
    return af.DynestyStatic(
        "name"
    )


def test_serialize_optimiser(
        optimizer
):
    optimizer = pickle.loads(
        pickle.dumps(
            optimizer
        )
    )
    assert optimizer.name == "name"


def test_serialize_grid_search(
        optimizer
):
    grid_search = af.SearchGridSearch(
        optimizer
    )
    assert grid_search.logger.name == "GridSearch (name)"
    assert "logger" not in grid_search.__getstate__()

    dumped = dill.dumps(
        grid_search
    )
    loaded = dill.loads(
        dumped
    )
    assert loaded.logger is not None
