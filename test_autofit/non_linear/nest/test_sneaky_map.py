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


@pytest.fixture(
    name="search"
)
def make_search():
    return af.DynestyStatic(
        number_of_cores=2
    )


@pytest.fixture(
    name="model"
)
def make_model():
    return af.Model(
        af.Gaussian
    )


@pytest.fixture(
    name="analysis"
)
def make_analysis():
    return Analysis()


@pytest.fixture(
    name="paths"
)
def make_paths(
        model,
        search
):
    paths = search.paths
    paths.model = model
    return model


class Analysis(af.Analysis):
    def log_likelihood_function(self, instance):
        return -1


class Fitness(af.NonLinearSearch.Fitness):
    def figure_of_merit_from(self, parameter_list):
        return -1


def times_two(x):
    return 2 * x


def test_sneaky_pool(
        model,
        analysis,
        paths
):
    number_of_cores = 2

    pool = SneakyPool(
        processes=number_of_cores,
        fitness=Fitness(
            paths=paths,
            model=model,
            analysis=analysis,
            samples_from_model=None
        )
    )
    ids = pool.map(times_two, range(number_of_cores))

    assert list(ids) == [0, 2]


def test_sneaky_map(
        search,
        model,
        analysis
):
    result = search.fit(
        model,
        analysis
    )

    assert isinstance(
        result.instance,
        af.Gaussian
    )
