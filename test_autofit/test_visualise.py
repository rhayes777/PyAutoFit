import pytest
from pathlib import Path

import autofit as af
from autofit.visualise import VisualiseGraph


@pytest.fixture
def model():
    return af.Model(af.Gaussian)


def test_visualise(model):
    graph_path = Path("graph.html")
    VisualiseGraph(model).save(str(graph_path))

    assert graph_path.exists()


def test_top_node():
    model = af.Model(af.Gaussian)
    graph = VisualiseGraph(model).graph()

    assert "Model(Gaussian)" in graph.nodes
