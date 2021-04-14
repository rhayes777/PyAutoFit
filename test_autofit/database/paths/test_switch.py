import autofit as af
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

    result = search.fit(
        af.Model(
            mock.Gaussian
        ),
        Analysis()
    )
