import pytest

import autofit as af
from autoconf.conf import with_config
from autofit.non_linear.analysis.multiprocessing import AnalysisPool
from autofit.non_linear.paths.abstract import AbstractPaths


class Analysis(af.Analysis):
    def __init__(self):
        self.did_visualise = False

    def log_likelihood_function(self, instance):
        return -1

    def visualize(
            self,
            paths: AbstractPaths,
            instance,
            during_analysis
    ):
        self.did_visualise = True


def test_visualise():
    analysis_1 = Analysis()
    analysis_2 = Analysis()

    (analysis_1 + analysis_2).visualize(
        af.DirectoryPaths(), None, None
    )

    assert analysis_1.did_visualise is True
    assert analysis_2.did_visualise is True

def test_make_result():

    analysis_1 = Analysis()
    analysis_2 = Analysis()

    result = (analysis_1 + analysis_2).make_result(
        samples=None, model=None, search=None
    )

    assert len(result) == 2

def test_add_analysis():
    assert (Analysis() + Analysis()).log_likelihood_function(
        None
    ) == -2


@pytest.mark.parametrize(
    "number, first, second",
    [
        (3, 2, 1),
        (4, 2, 2),
        (5, 3, 2),
        (6, 3, 3),
        (7, 4, 3),
    ]
)
def test_analysis_pool(
        number,
        first,
        second
):
    pool = AnalysisPool(
        number * [Analysis()],
        2
    )

    process_1, process_2 = pool.processes

    assert len(process_1.analyses) == first
    assert len(process_2.analyses) == second


# @with_config(
#     "general", "analysis", "n_cores",
#     value=2
# )
# @pytest.mark.parametrize(
#     "number",
#     list(range(1, 10))
# )
# def test_two_cores(number):
#     analysis = Analysis()
#     for _ in range(number - 1):
#         analysis += Analysis()
#     assert analysis.log_likelihood_function(
#         None
#     ) == -number


def test_still_flat():
    analysis = (Analysis() + Analysis()) + Analysis()

    assert len(analysis) == 3

    analysis = Analysis() + (Analysis() + Analysis())

    assert len(analysis) == 3
