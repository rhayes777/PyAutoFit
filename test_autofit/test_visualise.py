import itertools

import pytest
from pathlib import Path

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
def graph(visualise_graph):
    return visualise_graph.graph()


@pytest.mark.parametrize("node", ["0:Model(Gaussian)", "3:UniformPrior(0.0, 1.0)"])
def test_nodes(graph, node):
    assert node in graph.nodes


@pytest.mark.parametrize(
    "edge",
    [
        ("2:Collection(2)", "0:Model(Gaussian)"),
        ("2:Collection(2)", "1:Model(Gaussian)"),
        ("0:Model(Gaussian)", "1:UniformPrior(0.0, 1.0)"),
    ],
)
def test_edges(graph, edge):
    assert edge in graph.edges


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
