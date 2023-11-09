from pathlib import Path

from autoconf.conf import with_config


@with_config("general", "model", "ignore_prior_limits", value=True)
def test_sensitivity(sensitivity):
    results = sensitivity.run()
    assert len(results) == 8

    output_path = sensitivity.paths.output_path

    assert (output_path / ".is_grid_search").exists()
    path = output_path / "results.csv"
    assert path.exists()
    with open(path) as f:
        all_lines = set(f)
        assert (
            "index,centre,normalization,sigma,log_likelihood_base,log_likelihood_perturbed,log_likelihood_difference\n"
            in all_lines
        )
        assert "     0,  0.25,  0.25,  0.25,   2.0,   2.0,   0.0\n" in all_lines
        assert "     1,  0.25,  0.25,  0.75,   2.0,   2.0,   0.0\n" in all_lines
