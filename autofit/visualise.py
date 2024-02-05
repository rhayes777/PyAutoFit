import networkx as nx
from pyvis.network import Network

from autofit import (
    AbstractPriorModel,
    Model,
    Collection,
    UniformPrior,
    GaussianPrior,
    LogGaussianPrior,
    LogUniformPrior,
)


def str_for_object(obj):
    if isinstance(obj, Collection):
        return f"{obj.id}:Collection({len(obj)})"
    if isinstance(obj, Model):
        return f"{obj.id}:Model({obj.cls.__name__})"
    if isinstance(obj, UniformPrior):
        return f"{obj.id}:UniformPrior({obj.lower_limit}, {obj.upper_limit})"
    if isinstance(obj, GaussianPrior):
        return f"{obj.id}:GaussianPrior({obj.mean}, {obj.sigma})"
    if isinstance(obj, LogGaussianPrior):
        return f"{obj.id}:LogGaussianPrior({obj.mean}, {obj.sigma})"
    if isinstance(obj, LogUniformPrior):
        return f"{obj.id}:LogUniformPrior({obj.lower_limit}, {obj.upper_limit})"

    return repr(obj)


class VisualiseGraph:
    def __init__(self, model: AbstractPriorModel):
        self.model = model

    def graph(self):
        graph = nx.Graph()

        def add_model(model):
            model_name = str_for_object(model)
            graph.add_node(model_name)

            for name, prior in model.direct_prior_tuples:
                prior_name = str_for_object(prior)
                graph.add_node(prior_name)
                graph.add_edge(
                    model_name,
                    prior_name,
                    label=name,
                )

            for name, child_model in model.direct_prior_model_tuples:
                add_model(child_model)
                graph.add_edge(
                    model_name,
                    str_for_object(child_model),
                    label=name,
                )

        add_model(self.model)

        return graph

    def network(self, notebook: bool = False):
        net = Network(notebook=notebook)
        net.from_nx(self.graph())
        return net

    def save(self, path: str):
        self.network().save_graph(path)

    def show(self, name):
        self.network(notebook=True).show(f"{name}.html")
