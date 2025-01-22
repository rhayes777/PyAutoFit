import pytest

import autofit as af


def test_is_database_paths(session):
    mle = af.m.MockMLE(session=session)
    assert isinstance(mle.paths, af.DatabasePaths)
    # noinspection PyUnresolvedReferences
    assert mle.paths.save_all_samples is False


@pytest.mark.parametrize("save_all_samples", [True, False])
def test_save_all_samples_boolean(session, save_all_samples):
    mle = af.m.MockMLE(session=session, save_all_samples=save_all_samples)
    # noinspection PyUnresolvedReferences
    assert mle.paths.save_all_samples is save_all_samples


def test_unique_tag(session):
    analysis = af.m.MockAnalysis()
    model = af.Model(af.Gaussian)

    unique_tag = "unique"

    mle = af.m.MockMLE(session=session, unique_tag=unique_tag)

    assert mle.paths.unique_tag == unique_tag

    mle.fit(model, analysis)

    fit = session.query(af.db.Fit).one()

    assert fit.unique_tag == unique_tag
