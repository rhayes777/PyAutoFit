import pytest

import autofit as af


@pytest.fixture(name="paths")
def make_paths(session):
    return af.DatabasePaths(session)


def test_persist_database(samples, model, paths):
    paths.model = model
    paths.save_derived_quantities(samples)

    assert paths.fit["derived_quantities"].shape == (1, 1)


def test_custom_derived_quantities(session):
    model = af.Model(
        af.Gaussian,
        custom_derived_quantities={"custom": lambda instance: 2 * instance.centre},
    )

    obj = af.db.Object.from_object(model)

    assert "custom" in obj().custom_derived_quantities


def test_function(session):
    obj = af.db.Object.from_object(lambda x: x**2)
    assert obj()(2) == 4
