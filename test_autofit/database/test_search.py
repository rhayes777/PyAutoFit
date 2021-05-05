import pytest

import autofit as af
from autofit.mock.mock import MockAnalysis, Gaussian
from test_autofit.non_linear.grid.test_optimizer_grid_search import MockOptimizer


def test_is_database_paths(session):
    optimizer = MockOptimizer(
        session=session
    )
    assert isinstance(
        optimizer.paths,
        af.DatabasePaths
    )
    # noinspection PyUnresolvedReferences
    assert optimizer.paths.save_all_samples is False


@pytest.mark.parametrize(
    "save_all_samples",
    [True, False]
)
def test_save_all_samples_boolean(
        session,
        save_all_samples
):
    optimizer = MockOptimizer(
        session=session,
        save_all_samples=save_all_samples
    )
    # noinspection PyUnresolvedReferences
    assert optimizer.paths.save_all_samples is save_all_samples


@pytest.mark.parametrize(
    "save_all_samples, n_samples",
    [
        (False, 1),
        (True, 2)
    ]
)
def test_save_all_samples(
        session,
        save_all_samples,
        n_samples
):
    analysis = MockAnalysis()
    model = af.Model(Gaussian)

    optimizer = MockOptimizer(
        session=session,
        save_all_samples=save_all_samples,
        sample_multiplier=2
    )

    optimizer.fit(
        model,
        analysis
    )

    fit = session.query(
        af.db.Fit
    ).one()

    assert len(fit.samples.samples) == n_samples


def test_unique_tag(session):
    analysis = MockAnalysis()
    model = af.Model(Gaussian)

    unique_tag = "unique"

    optimizer = MockOptimizer(
        session=session,
        unique_tag=unique_tag
    )

    assert optimizer.paths.unique_tag == unique_tag

    optimizer.fit(
        model,
        analysis
    )

    fit = session.query(
        af.db.Fit
    ).one()

    assert fit.unique_tag == unique_tag
