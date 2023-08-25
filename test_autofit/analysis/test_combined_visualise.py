import pytest

import autofit as af


class Analysis(af.Analysis):
    def visualize(self, paths, instance, during_analysis):
        assert isinstance(instance, af.Gaussian)
        assert during_analysis is True

        paths.output_path.mkdir(parents=True, exist_ok=True)
        with open(f"{paths.output_path}/test.txt", "w+") as f:
            f.write("test")


@pytest.fixture(name="analysis")
def make_analysis():
    return Analysis()


@pytest.fixture(name="paths")
def make_paths():
    return af.DirectoryPaths()


def test_visualize(analysis, paths):
    analysis.visualize(paths, af.Gaussian(), True)

    assert (paths.output_path / "test.txt").exists()


@pytest.fixture(name="combined")
def make_combined(analysis):
    combined = analysis + analysis
    combined.n_cores = 2
    yield combined
    combined._analysis_pool.terminate()


def test_combined_visualize(combined, paths):
    combined.visualize(paths, af.Gaussian(), True)

    analyses_path = paths.output_path / "analyses"

    assert (analyses_path / "analysis_0/test.txt").exists()
    assert (analyses_path / "analysis_1/test.txt").exists()
