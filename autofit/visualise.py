import os

import networkx as nx
from pyvis.network import Network

from autofit import AbstractPriorModel


class VisualiseGraph:
    def __init__(self, model: AbstractPriorModel):
        self.model = model

    def graph(self):
        # Create a graph object
        G = nx.Graph()

        # Add nodes
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")

        # Add edges
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "A")

        return G

    def save(self, path: str):
        net = Network()
        net.from_nx(self.graph())
        net.save_graph(path)

    def show(self, name):
        net = Network()
        net.from_nx(self.graph())
        net.show(f"{name}.html")
