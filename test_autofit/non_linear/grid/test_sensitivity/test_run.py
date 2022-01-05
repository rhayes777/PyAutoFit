from pathlib import Path


def test_sensitivity(
        sensitivity
):
    results = sensitivity.run()
    assert len(results) == 8

    path = Path(
        sensitivity.search.paths.output_path
    ) / "results.csv"
    assert path.exists()
    with open(path) as f:
        assert next(
            f) == 'index,centre,normalization,sigma,log_likelihood_base,log_likelihood_perturbed,log_likelihood_difference\n'
        assert next(f) == '0,0.25,0.25,0.25,2.0,2.0,0.0\n'
        assert next(f) == '1,0.25,0.25,0.75,2.0,2.0,0.0\n'
