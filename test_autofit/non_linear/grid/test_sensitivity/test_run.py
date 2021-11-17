from pathlib import Path

import pytest


@pytest.fixture(
    autouse=True,
    name="results"
)
def run_sensitivity(
        sensitivity
):
    return sensitivity.run()


def test_sensitivity(
        results,
        sensitivity
):
    assert len(results) == 8

    path = Path(
        sensitivity.search.paths.output_path
    ) / "results.csv"
    assert path.exists()
    with open(path) as f:
        assert next(f) == 'index,centre,intensity,sigma,log_likelihood_difference\n'
        assert next(f) == '0,0.25,0.25,0.25,0.0\n'
        assert next(f) == '1,0.25,0.25,0.75,0.0\n'
