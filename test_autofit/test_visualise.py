import itertools

import pytest

import autofit as af
from autofit.visualise import VisualiseGraph


@pytest.fixture(autouse=True)
def reset_ids():
    af.ModelObject._ids = itertools.count()
    af.Prior._ids = itertools.count()


@pytest.fixture
def model():
    collection = af.Collection(
        first=af.Model(af.Gaussian),
        second=af.Model(af.Gaussian),
    )
    collection.first.centre = collection.second.centre
    return collection


@pytest.fixture
def visualise_graph(model):
    return VisualiseGraph(model)


def test_visualise(model, visualise_graph, graph_path):
    visualise_graph.save(str(graph_path))

    assert graph_path.exists()


@pytest.fixture
def graph_path(output_directory):
    return output_directory / "graph.html"


def test_complex_object(graph_path):
    second = af.Model(af.Gaussian)
    exp = af.Model(af.Exponential)
    third = af.Model(af.Gaussian)

    collection = af.Collection(
        first=af.Collection(
            second=second,
            exp=exp,
        ),
        third=third,
    )
    third.centres = exp.centre
    second.sigma = third.sigma

    visualise_graph = VisualiseGraph(collection)
    visualise_graph.save(str(graph_path))

    assert graph_path.exists()
