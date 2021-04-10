import autofit as af


def test_dynesty(session):
    search = af.DynestyStatic(
        session=session
    )
    assert isinstance(
        search.paths,
        af.DatabasePaths
    )
