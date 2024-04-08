import os
from pathlib import Path

import pytest

import autofit as af
from autoconf.conf import with_config
from autofit.non_linear.analysis.multiprocessing import AnalysisPool
from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.non_linear.paths.sub_directory_paths import SubDirectoryPaths


class Analysis(af.Analysis):
    def __init__(self):
        self.did_visualise = False
        self.did_visualise_combined = False
        self.did_profile = False

    def log_likelihood_function(self, instance):
        return -1

    def visualize_before_fit(self, paths: AbstractPaths, model):
        self.did_visualise = True
        os.makedirs(paths.image_path, exist_ok=True)
        open(f"{paths.image_path}/image.png", "w+").close()

    def visualize(self, paths: AbstractPaths, instance, during_analysis):
        self.did_visualise = True
        os.makedirs(paths.image_path, exist_ok=True)
        open(f"{paths.image_path}/image.png", "w+").close()

    def visualize_before_fit_combined(self, analyses, paths, model):
        self.did_visualise_combined = True

    def visualize_combined(
        self, analyses, paths: AbstractPaths, instance, during_analysis
    ):
        self.did_visualise_combined = True

    def profile_log_likelihood_function(self, paths: AbstractPaths, instance):
        self.did_profile = True


def test_visualise_before_fit():
    analysis_1 = Analysis()
    analysis_2 = Analysis()

    (analysis_1 + analysis_2).visualize_before_fit(af.DirectoryPaths(), None)

    assert analysis_1.did_visualise is True
    assert analysis_2.did_visualise is True


def test_visualise():
    analysis_1 = Analysis()
    analysis_2 = Analysis()

    (analysis_1 + analysis_2).visualize(af.DirectoryPaths(), None, None)

    assert analysis_1.did_visualise is True
    assert analysis_2.did_visualise is True


def test_visualise_before_fit_combined():
    analysis_1 = Analysis()
    analysis_2 = Analysis()

    (analysis_1 + analysis_2).visualize_before_fit_combined(
        None, af.DirectoryPaths(), None
    )

    assert analysis_1.did_visualise_combined is True
    assert analysis_2.did_visualise_combined is False


def test_visualise_combined():
    analysis_1 = Analysis()
    analysis_2 = Analysis()

    (analysis_1 + analysis_2).visualize_combined(None, af.DirectoryPaths(), None, None)

    assert analysis_1.did_visualise_combined is True
    assert analysis_2.did_visualise_combined is False


def test__profile_log_likelihood():
    analysis_1 = Analysis()
    analysis_2 = Analysis()

    (analysis_1 + analysis_2).profile_log_likelihood_function(
        af.DirectoryPaths(),
        None,
    )

    assert analysis_1.did_profile is True
    assert analysis_2.did_profile is True


def test_make_result():
    analysis_1 = Analysis()
    analysis_2 = Analysis()

    result = (analysis_1 + analysis_2).make_result(
        samples_summary=None,
        paths=None,
        samples=None,
    )

    assert len(result) == 2


def test_add_analysis():
    assert (Analysis() + Analysis()).log_likelihood_function(None) == -2


@pytest.mark.parametrize(
    "number, first, second",
    [
        (3, 2, 1),
        (4, 2, 2),
        (5, 3, 2),
        (6, 3, 3),
        (7, 4, 3),
    ],
)
def test_analysis_pool(number, first, second):
    pool = AnalysisPool(number * [Analysis()], 2)

    process_1, process_2 = pool.processes

    assert len(process_1.analyses) == first
    assert len(process_2.analyses) == second


@with_config("general", "analysis", "n_cores", value=2)
@pytest.mark.parametrize("number", list(range(1, 3)))
def test_two_cores(number):
    analysis = Analysis()
    for _ in range(number - 1):
        analysis += Analysis()
    assert analysis.log_likelihood_function(None) == -number


def test_still_flat():
    analysis = (Analysis() + Analysis()) + Analysis()

    assert len(analysis) == 3

    analysis = Analysis() + (Analysis() + Analysis())

    assert len(analysis) == 3


def test_sum_analyses():
    combined = sum(Analysis() for _ in range(5))
    assert len(combined) == 5


@pytest.fixture(name="search")
def make_search():
    return af.m.MockSearch("search_name")


def test_child_paths(search):
    paths = search.paths
    sub_paths = SubDirectoryPaths(paths, analysis_name="analysis_0")
    assert sub_paths.output_path == paths.output_path / "analysis_0"


@pytest.fixture(name="multi_analysis")
def make_multi_analysis():
    return Analysis() + Analysis()


@pytest.fixture(name="multi_search")
def make_multi_search(search, multi_analysis):
    search.paths.remove_files = False

    search.fit(af.Model(af.Gaussian), multi_analysis)
    search.paths.save_all({}, {})
    return search


@with_config("general", "output", "remove_files", value=False)
def test_visualise(multi_search, multi_analysis):
    multi_analysis.visualize(multi_search.paths, af.Gaussian(), True)
    search_path = Path(multi_search.paths.output_path)
    assert search_path.exists()
    assert (search_path / "analyses/analysis_0/image/image.png").exists()
    assert (search_path / "analyses/analysis_1/image/image.png").exists()


def test_set_number_of_cores(multi_analysis):
    multi_analysis.n_cores = 1
    assert multi_analysis._log_likelihood_function.__name__ == "_summed_log_likelihood"

    multi_analysis.n_cores = 2
    assert isinstance(multi_analysis._log_likelihood_function, AnalysisPool)
    multi_analysis.n_cores = 1
    assert multi_analysis._log_likelihood_function.__name__ == "_summed_log_likelihood"
