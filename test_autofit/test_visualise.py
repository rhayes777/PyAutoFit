import itertools

import pytest
from pathlib import Path

import autofit as af
from autofit.visualise import VisualiseGraph


@pytest.fixture(autouse=True)
def reset_ids():
    af.Prior._ids = itertools.count()


@pytest.fixture
def model():
    return af.Model(af.Gaussian)


@pytest.fixture
def visualise_graph(model):
    return VisualiseGraph(model)


def test_visualise(model, visualise_graph):
    graph_path = Path("graph.html")
    visualise_graph.save(str(graph_path))

    assert graph_path.exists()


@pytest.fixture
def graph(visualise_graph):
    return visualise_graph.graph()


@pytest.mark.parametrize("node", ["Model(Gaussian)", "UniformPrior_0(0.0, 1.0)"])
def test_nodes(graph, node):
    assert node in graph.nodes


@pytest.mark.parametrize("edge", [("Model(Gaussian)", "UniformPrior_0(0.0, 1.0)")])
def test_edges(graph, edge):
    assert edge in graph.edges
