"""
The graphical pass -- plate notation over a ``FactorGraphModel``.

The load-bearing assertion in this file is that **a draw is not sharing**: the
drawn ``Prior`` *is* the dataset model's own parameter object, so an extraction
that does nothing about it reports the three datasets as sharing one ``centre``
-- the exact opposite of what a hierarchical model says.  Everything else here
(hoisted hyper nodes, observed rows, the dataset plate, the split counts) exists
to make that statement drawable.
"""

import itertools
import json
import subprocess
import sys

import autofit as af
from autofit.graph_spec import GraphSpec
from autofit.tools.namer import namer

from .conftest import NullAnalysis
from .graphical_doubles import (
    dataset_model,
    hierarchical_graph,
    relational_graph,
    shared_graph,
    variable_graph,
)


def _reset_ids():
    af.ModelObject._ids = itertools.count()
    af.Prior._ids = itertools.count()
    namer.reset()


# -- the hierarchical model -------------------------------------------------


def test_one_hyper_node_is_hoisted_to_the_front():
    spec = GraphSpec.from_model(hierarchical_graph().global_prior_model)

    hyper = [child for child in spec.root.children if child.kind == "hyper"]
    assert len(hyper) == 1
    # Hoisted: the one documented exception to declaration order.
    assert spec.root.children[0] is hyper[0]

    node = hyper[0]
    assert node.path == ("HierarchicalFactor0",)
    assert node.cls_name == "GaussianPrior"
    assert node.factor.kind == "hierarchical"
    assert [row.name for row in node.rows] == ["mean", "sigma"]
    # The per-draw `_HierarchicalFactor` collections are gone: one node stands
    # for the one distribution they all point at.
    assert [child.name for child in spec.root.children] == ["HierarchicalFactor0", "0"]


def test_a_draw_is_not_sharing():
    spec = GraphSpec.from_model(hierarchical_graph().global_prior_model)

    row = spec.row(("0", "centre"))
    assert row.provenance.kind == "hierarchical-draw"
    assert row.provenance.expression == "~ GaussianPrior(mean, sigma)"
    assert row.provenance.operands == ("HierarchicalFactor0",)
    # Still sampled: a draw is a provenance, never a sampling state.
    assert row.sampling == "free"
    # ... and never a sharing marker.
    assert row.shared is False
    assert row.direct_occurrences == (("0", "centre"),)
    assert spec.shared == ()
    assert spec.counts["shared_priors"] == 0


def test_one_draw_edge_per_dataset_in_dataset_order():
    spec = GraphSpec.from_model(hierarchical_graph().global_prior_model)

    assert len(spec.draws) == 3
    assert [edge.target_path for edge in spec.draws] == [
        ("0", "centre"),
        ("1", "centre"),
        ("2", "centre"),
    ]
    assert {edge.source_path for edge in spec.draws} == {("HierarchicalFactor0",)}
    assert len({edge.prior_id for edge in spec.draws}) == 3
    for edge in spec.draws:
        assert edge.expression == "~ GaussianPrior(mean, sigma)"


def test_the_dataset_plate_still_forms_around_a_draw():
    spec = GraphSpec.from_model(hierarchical_graph().global_prior_model)

    plate = spec.root.children[1]
    assert plate.plate.count == 3
    assert plate.plate.representative_key == "0 - 2"
    assert plate.factor.kind == "analysis"
    # Nothing is shared in all, because nothing is shared at all.
    assert plate.plate.shared_in_all == ()
    assert "centre ◂ drawn" in plate.plate.repeats[0]


def test_hierarchical_counts_reconcile():
    spec = GraphSpec.from_model(hierarchical_graph().global_prior_model)

    counts = spec.counts
    assert counts["datasets"] == 3
    assert counts["hyper_parameters"] == 2
    assert counts["shared_across_datasets"] == 0
    assert counts["per_dataset"] == 3
    assert counts["observed"] == 3
    assert (
        counts["hyper_parameters"]
        + counts["shared_across_datasets"]
        + counts["per_dataset"] * counts["datasets"]
        == counts["unique_sampled_scalars"]
    )


def test_the_hyper_node_records_the_model_info_paths_it_replaces():
    spec = GraphSpec.from_model(hierarchical_graph().global_prior_model)

    entry = spec.path_index["HierarchicalFactor0"]
    paths = entry["figure"] if isinstance(entry, dict) else [
        "/".join(path) for path in entry
    ]
    # The three `_HierarchicalFactor` collections, grouped as `model.info`
    # groups them -- recorded rather than hidden.
    assert paths and all("distribution_model" in path for path in paths)


# -- the shared model -------------------------------------------------------


def test_shared_model_collapses_to_one_plate_of_three():
    model = shared_graph().global_prior_model
    spec = GraphSpec.from_model(model)

    assert len(spec.root.children) == 1
    plate = spec.root.children[0]
    assert plate.plate.count == 3
    assert plate.plate.member_paths == (("0",), ("1",), ("2",))
    assert len(plate.plate.shared_in_all) == 3
    assert {row.prior_id for row in plate.rows if row.prior_id is not None} == set(
        plate.plate.shared_in_all
    )
    assert spec.draws == ()

    counts = spec.counts
    assert counts["datasets"] == 3
    assert counts["shared_across_datasets"] == 3
    assert counts["per_dataset"] == 0
    assert counts["hyper_parameters"] == 0


def test_variable_model_shares_only_the_centre():
    spec = GraphSpec.from_model(variable_graph().global_prior_model)

    counts = spec.counts
    assert counts["shared_across_datasets"] == 1
    assert counts["per_dataset"] == 2
    assert (
        counts["shared_across_datasets"]
        + counts["per_dataset"] * counts["datasets"]
        == counts["unique_sampled_scalars"]
    )


def test_a_relation_across_datasets_counts_as_shared_across_datasets():
    """``sigma_m`` / ``sigma_c`` are one pair of numbers for every dataset."""
    spec = GraphSpec.from_model(relational_graph().global_prior_model)

    counts = spec.counts
    assert counts["shared_across_datasets"] == 4
    assert counts["per_dataset"] == 0
    assert counts["unique_sampled_scalars"] == 4


# -- observed data ----------------------------------------------------------


def test_observed_rows_come_from_the_analysis():
    spec = GraphSpec.from_model(shared_graph().global_prior_model, collapse=False)

    for index in ("0", "1", "2"):
        row = spec.row((index, "data"))
        assert row.sampling == "observed"
        assert row.provenance.kind == "observed"
        # Not part of the model, so not part of `model.info`.
        assert row.in_model_info is False
        assert spec.path_index[f"{index}/data"] == ()
    assert spec.counts["observed"] == 3
    # An observed node is not a fixed constant.
    assert spec.counts["fixed_leaf_slots"] == 0


def test_an_analysis_without_data_contributes_no_observed_row():
    spec = GraphSpec.from_model(
        shared_graph(analysis_cls=lambda index: NullAnalysis()).global_prior_model
    )

    assert spec.counts["observed"] == 0
    assert all(
        row.provenance.kind != "observed"
        for node in spec.components()
        for row in node.rows
    )
    # The factor is still recorded.
    assert spec.root.children[0].factor.kind == "analysis"


# -- the factor graph is read once ------------------------------------------


def test_the_factor_graph_is_read_exactly_once():
    """
    ``FactorGraphModel.graph`` rebuilds and renames its ``PriorFactor``s on every
    access, so a second read would silently change the names the figure quotes.
    """
    graph_model = hierarchical_graph()
    model = graph_model.global_prior_model

    reads = []
    inherited = type(graph_model).graph

    def _counting(self):
        reads.append(self)
        return inherited.fget(self)

    type(graph_model).graph = property(_counting)
    try:
        spec = GraphSpec.from_model(model)
    finally:
        del type(graph_model).graph

    assert len(reads) == 1
    assert len(spec.draws) == 3


# -- nothing changes for a model that is not a factor graph -----------------


def test_a_plain_model_is_untouched_by_the_graphical_pass():
    spec = GraphSpec.from_model(af.Collection(one=dataset_model(), two=dataset_model()))

    assert spec.draws == ()
    assert all(node.factor is None for node in spec.components())
    assert all(node.kind != "hyper" for node in spec.components())
    for key in ("datasets", "hyper_parameters", "shared_across_datasets"):
        assert key not in spec.counts


# -- determinism ------------------------------------------------------------


def test_graphical_extraction_is_deterministic():
    first = json.dumps(GraphSpec.from_model(hierarchical_graph().global_prior_model).to_dict())
    _reset_ids()
    second = json.dumps(GraphSpec.from_model(hierarchical_graph().global_prior_model).to_dict())

    assert first == second


def test_graphical_extraction_is_deterministic_in_a_fresh_process():
    source = (
        "import json, sys;"
        "sys.path.insert(0, %r);"
        "from test_autofit.graph_spec.graphical_doubles import hierarchical_graph;"
        "from autofit.graph_spec import GraphSpec;"
        "print(json.dumps(GraphSpec.from_model("
        "hierarchical_graph().global_prior_model).to_dict()))"
    ) % str(_repo_root())
    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=True
    )

    assert json.loads(result.stdout) == json.loads(
        json.dumps(GraphSpec.from_model(hierarchical_graph().global_prior_model).to_dict())
    )


def _repo_root():
    from pathlib import Path

    return Path(__file__).resolve().parents[2]
