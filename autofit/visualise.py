import networkx as nx
from pyvis.network import Network

from autoconf import cached_property
from autofit import (
    AbstractPriorModel,
    Model,
    Collection,
    UniformPrior,
    GaussianPrior,
    LogGaussianPrior,
    LogUniformPrior,
)
import colorsys


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


def generate_n_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        r, g, b = colorsys.hls_to_rgb(hue, 0.5, 1.0)
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(r * 255), int(g * 255), int(b * 255)
        )
        colors.append(hex_color)
    return colors


class VisualiseGraph:
    def __init__(self, model: AbstractPriorModel):
        self.model = model

    def graph(self):
        graph = nx.DiGraph()

        def add_model(model):
            model_name = str_for_object(model)

            for name, prior in model.direct_prior_tuples:
                prior_name = str_for_object(prior)
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

    @cached_property
    def colours(self):
        types = {
            Collection,
            UniformPrior,
            GaussianPrior,
            LogGaussianPrior,
            LogUniformPrior,
        } | {model.cls for _, model in self.model.attribute_tuples_with_type(Model)}
        if isinstance(self.model, Model):
            types.add(self.model.cls)
        return {
            type_: color
            for type_, color in zip(
                sorted(types, key=str),
                generate_n_colors(len(types)),
            )
        }

    def network(self, notebook: bool = False):
        net = Network(
            notebook=notebook,
            directed=True,
        )

        def add_model(obj):
            net.add_node(
                str_for_object(obj),
                shape="square",
                color=self.colours[model.cls],
                size=15,
            )

        def add_collection(obj):
            net.add_node(
                str_for_object(obj),
                shape="hexagon",
                color=self.colours[Collection],
                size=15,
            )

        if isinstance(self.model, Model):
            add_model(self.model)
        else:
            add_collection(self.model)

        for _, model in self.model.attribute_tuples_with_type(Model):
            add_model(model)
        for _, collection in self.model.attribute_tuples_with_type(Collection):
            add_collection(collection)
        for _, prior in self.model.prior_tuples:
            net.add_node(
                str_for_object(prior),
                shape="dot",
                color=self.colours[type(prior)],
                size=10,
            )

        net.from_nx(self.graph())

        return net

    def save(self, path: str):
        self.network().save_graph(path)

    def show(self, name):
        self.network(notebook=True).show(f"{name}.html")
