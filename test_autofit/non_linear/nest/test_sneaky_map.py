import shutil
from pathlib import Path

import pytest

import autofit as af
from autoconf import conf
from autofit.non_linear.parallel import SneakyPool


@pytest.fixture(
    autouse=True
)
def manage_output_path():
    original_path = conf.instance.output_path
    new_output_path = str(
        Path(__file__).parent / "output"
    )
    conf.instance.output_path = new_output_path
    yield
    shutil.rmtree(
        new_output_path,
        ignore_errors=True
    )
    conf.instance.output_path = original_path


class Analysis(af.Analysis):
    def log_likelihood_function(self, instance):
        return -1


def times_two(x):
    return 2 * x


def test_sneaky_pool():
    number_of_cores = 2

    pool = SneakyPool(
        processes=number_of_cores
    )
    ids = pool.map(times_two, range(number_of_cores))

    assert list(ids) == [0, 2]


def test_sneaky_map():
    search = af.DynestyStatic(
        number_of_cores=2
    )
    model = af.Model(
        af.Gaussian
    )

    analysis = Analysis()

    result = search.fit(
        model,
        analysis
    )

    assert isinstance(
        result.instance,
        af.Gaussian
    )
