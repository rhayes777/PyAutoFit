from autofit import DatabasePaths


def test_persist_database(samples, model, session):
    paths = DatabasePaths(session)
    paths.model = model
    paths.save_derived_quantities(samples)

    assert paths.fit["derived_quantities"].shape == (1, 1)
