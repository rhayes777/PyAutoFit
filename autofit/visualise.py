import networkx as nx
from pyvis.network import Network

from autofit import (
    AbstractPriorModel,
    Model,
    UniformPrior,
    GaussianPrior,
    LogGaussianPrior,
    LogUniformPrior,
)


def str_for_object(obj):
    if isinstance(obj, Model):
        return f"Model({obj.cls.__name__})"
    if isinstance(obj, UniformPrior):
        return f"UniformPrior_{obj.id}({obj.lower_limit}, {obj.upper_limit})"
    if isinstance(obj, GaussianPrior):
        return f"GaussianPrior_{obj.id}({obj.mean}, {obj.sigma})"
    if isinstance(obj, LogGaussianPrior):
        return f"LogGaussianPrior_{obj.id}({obj.mean}, {obj.sigma})"
    if isinstance(obj, LogUniformPrior):
        return f"LogUniformPrior_{obj.id}({obj.lower_limit}, {obj.upper_limit})"

    return repr(obj)


class VisualiseGraph:
    def __init__(self, model: AbstractPriorModel):
        self.model = model

    def graph(self):
        graph = nx.Graph()

        graph.add_node(str_for_object(self.model))

        for prior in self.model.priors:
            graph.add_node(str_for_object(prior))

        # # Add nodes
        # G.add_node("A")
        # G.add_node("B")
        # G.add_node("C")
        #
        # # Add edges
        # G.add_edge("A", "B")
        # G.add_edge("B", "C")
        # G.add_edge("C", "A")

        return graph

    def save(self, path: str):
        net = Network()
        net.from_nx(self.graph())
        net.save_graph(path)

    def show(self, name):
        net = Network()
        net.from_nx(self.graph())
        net.show(f"{name}.html")
