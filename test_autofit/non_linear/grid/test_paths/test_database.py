import autofit as af


def test_paths_type(database_paths):
    for path in database_paths:
        assert isinstance(path, af.DatabasePaths)


def test_name_prefix_tag(session):
    paths = af.DatabasePaths(
        session, name="name", path_prefix="prefix", unique_tag="tag"
    )

    fit = paths.fit
    assert fit.name == "name"
    assert fit.path_prefix == "prefix"
    assert fit.unique_tag == "tag"
