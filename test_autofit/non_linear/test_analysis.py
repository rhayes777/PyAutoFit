import pytest

import autofit as af
from autoconf.conf import with_config
from autofit.non_linear.analysis.multiprocessing import AnalysisPool


class Analysis(af.Analysis):
    def log_likelihood_function(self, instance):
        return -1


def test_add_analysis():
    assert (Analysis() + Analysis()).log_likelihood_function(
        None
    ) == -2


def test_analysis_pool():
    pool = AnalysisPool(
        3 * [Analysis()],
        2
    )

    process_1, process_2 = pool.processes

    assert len(process_1.analyses) == 2
    assert len(process_2.analyses) == 1


@with_config(
    "general", "analysis", "n_cores",
    value=2
)
@pytest.mark.parametrize(
    "number",
    list(range(1, 10))
)
def test_two_cores(number):
    analysis = Analysis()
    for _ in range(number - 1):
        analysis += Analysis()
    assert analysis.log_likelihood_function(
        None
    ) == -number


def test_still_flat():
    analysis = (Analysis() + Analysis()) + Analysis()

    assert len(analysis) == 3

    analysis = Analysis() + (Analysis() + Analysis())

    assert len(analysis) == 3
