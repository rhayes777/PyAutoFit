import autofit as af
from autofit import database as m
from autofit.mapper.model_object import Identifier
from autofit.mock import mock


class Analysis(af.Analysis):

    def log_likelihood_function(self, instance):
        return -1


def test_dynesty(session):
    search = af.DynestyStatic(
        session=session
    )
    assert isinstance(
        search.paths,
        af.DatabasePaths
    )

    model = af.Model(
        mock.Gaussian
    )

    print("fit")

    search.fit(
        model,
        Analysis()
    )

    fit, = m.Fit.all(session)

    print("first identifier check")
    assert fit.id == search.paths.identifier
    assert fit.is_complete

    search = af.DynestyStatic(
        session=session
    )

    search.paths.model = model

    print("second identifier check")
    assert search.paths.identifier == fit.id
    assert search.paths.is_complete

    search.fit(
        model,
        Analysis()
    )
