import shutil
from pathlib import Path

import pytest

import autofit as af
from autoconf import conf
from autofit.non_linear.parallel import SneakyPool, SneakyJob


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
        number_of_cores=2,
        maxcall=10
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


@pytest.fixture(
    name="fitness"
)
def make_fitness(
        paths,
        model,
        analysis,
):
    return Fitness(
        paths=paths,
        model=model,
        analysis=analysis,
        samples_from_model=None
    )


class Analysis(af.Analysis):
    def log_likelihood_function(self, instance):
        return -1


class Fitness(af.NonLinearSearch.Fitness):
    def figure_of_merit_from(self, parameter_list):
        return -1


@pytest.fixture(
    name="sneaky_job"
)
def make_sneaky_job(
        fitness
):
    return SneakyJob(
        identity,
        1,
        fitness
    )


def test_sneaky_removal(
        sneaky_job
):
    assert sneaky_job.args == [1]
    assert sneaky_job.fitness_index == 1


def test_sneaky_call(
        sneaky_job,
        fitness
):
    assert sneaky_job.perform(
        fitness
    ) == ([1, fitness],)


def identity(*args):
    return args


def test_sneaky_pool(
        fitness
):
    number_of_cores = 2

    pool = SneakyPool(
        processes=number_of_cores,
        fitness=fitness
    )
    results = list(pool.map(identity, [(0, fitness), (1, fitness)]))

    assert len(results) == 2

    for i, t in enumerate(results):
        assert t[0][0] == i
        assert isinstance(
            t[0][1],
            Fitness
        )


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
