import autofit as af


def test_save_instance(
        session
):
    paths = af.DatabasePaths(
        session=session
    )
    paths.save_named_instance(
        "name",
        af.Gaussian()
    )
    assert isinstance(
        paths.fit.named_instances[
            "name"
        ],
        af.Gaussian

    )


def test_paths_type(
        database_paths
):
    for path in database_paths:
        assert isinstance(
            path,
            af.DatabasePaths
        )


def test_name_prefix_tag(
        session
):
    paths = af.DatabasePaths(
        session,
        name="name",
        path_prefix="prefix",
        unique_tag="tag"
    )

    fit = paths.fit
    assert fit.name == "name"
    assert fit.path_prefix == "prefix"
    assert fit.unique_tag == "tag"
