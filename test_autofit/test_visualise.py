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
    return af.Collection(
        first=af.Model(af.Gaussian),
        second=af.Model(af.Gaussian),
    )


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


@pytest.mark.parametrize("node", ["0:Model(Gaussian)", "3:UniformPrior(0.0, 1.0)"])
def test_nodes(graph, node):
    assert node in graph.nodes


@pytest.mark.parametrize(
    "edge",
    [
        ("2:Collection(2)", "0:Model(Gaussian)"),
        ("2:Collection(2)", "1:Model(Gaussian)"),
        ("0:Model(Gaussian)", "0:UniformPrior(0.0, 1.0)"),
        ("0:Model(Gaussian)", "1:UniformPrior(0.0, 1.0)"),
    ],
)
def test_edges(graph, edge):
    assert edge in graph.edges
