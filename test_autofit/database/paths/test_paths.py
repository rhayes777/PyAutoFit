import autofit as af
from autofit import database as m


def test_complete(
        session
):
    paths = af.DatabasePaths(
        session
    )
    assert paths.is_complete is False

    fit, = m.Fit.all(session)
    assert fit.id == paths.identifier
