import pytest

import autofit as af


class Analysis(af.Analysis):
    def visualize(self, paths, instance, during_analysis):
        assert isinstance(instance, af.Gaussian)
        assert during_analysis is True

        paths.output_path.mkdir(parents=True, exist_ok=True)
        with open(f"{paths.output_path}/visualize.txt", "w+") as f:
            f.write("test")

    def visualize_before_fit(self, paths, model):
        assert model.cls is af.Gaussian

        paths.output_path.mkdir(parents=True, exist_ok=True)
        with open(f"{paths.output_path}/visualize_before_fit.txt", "w+") as f:
            f.write("test")


@pytest.fixture(name="analysis")
def make_analysis():
    return Analysis()


@pytest.fixture(name="paths")
def make_paths():
    return af.DirectoryPaths()


def test_visualize(analysis, paths):
    analysis.visualize(paths, af.Gaussian(), True)

    assert (paths.output_path / "visualize.txt").exists()


@pytest.fixture(name="combined")
def make_combined(analysis):
    combined = analysis + analysis
    combined.n_cores = 2
    yield combined
    combined._analysis_pool.terminate()


@pytest.fixture(name="analyses_path")
def make_analyses_path(paths):
    return paths.output_path / "analyses"


def test_combined_visualize(
    combined,
    paths,
    analyses_path,
):
    combined.visualize(
        paths,
        af.Gaussian(),
        True,
    )

    assert (analyses_path / "analysis_0/visualize.txt").exists()
    assert (analyses_path / "analysis_1/visualize.txt").exists()


def test_visualize_before_fit(
    combined,
    paths,
    analyses_path,
):
    combined.visualize_before_fit(
        paths,
        af.Model(af.Gaussian),
    )

    assert (analyses_path / "analysis_0/visualize_before_fit.txt").exists()
    assert (analyses_path / "analysis_1/visualize_before_fit.txt").exists()
