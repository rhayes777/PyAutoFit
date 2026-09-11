"""
Plate notation -- the presentation of a ``FactorGraphModel``.

The acceptance of the epic's phase 4, asserted on the figure:

* a hierarchical model's figure shows ``centre · drawn`` with the parent arrow
  landing **on the pill**, and **no** shared marker anywhere on it;
* a shared model's figure hoists the shared priors above the plate;
* observed data is visually distinct from a fixed constant;
* the footer separates hyper-parameters from the per-dataset parameters;
* and a model that is *not* a factor graph is presented exactly as before.
"""

import hashlib
import itertools
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import autofit as af
from autofit.graph_spec import GraphSpec
from autofit.model_figure.presentation import build_presentation, prior_summaries
from autofit.model_figure.render import PALETTE, _PILL_STYLE, _badge_colours

from test_autofit.graph_spec.graphical_doubles import (
    hierarchical_graph,
    relational_graph,
    shared_graph,
    variable_graph,
)
from test_autofit.graph_spec.lens_doubles import (
    group_model,
    mge_model,
    simple_lens_model,
)


def _presentation(model, **kwargs):
    return af.ModelPlotter(model).presentation(**kwargs)


def _pill(presentation, key):
    for pill in presentation.pills():
        if pill.key == key:
            return pill
    raise AssertionError(f"no pill at {key!r}")


def _card(presentation, key):
    for card in presentation.walk():
        if card.key == key:
            return card
    raise AssertionError(f"no card at {key!r}")


# -- shared: the prior is drawn once, above the plate -----------------------


def test_a_shared_model_hoists_its_priors_above_the_plate():
    presentation = _presentation(shared_graph().global_prior_model)

    hoist = presentation.cards[0]
    assert hoist.kind == "hoist"
    assert hoist.title == "shared across datasets"
    assert hoist.subtitle == "one value across 3 datasets"
    assert [pill.text for pill in hoist.pills] == ["centre", "normalization", "sigma"]

    plate = presentation.cards[1]
    assert plate.kind == "plate"
    assert plate.badge == "3 datasets"
    assert plate.subtitle == "dataset 0 - 2"

    # Every member points at the hoisted pill, and nowhere else.
    for name in ("centre", "normalization", "sigma"):
        assert _pill(presentation, f"0/{name}").badge == "↗ shared"
        assert (
            af.model_figure.presentation.Link(
                f"0/{name}", f"shared across datasets/{name}", "shared"
            )
            in presentation.links
        )


def test_every_occurrence_points_at_the_hoisted_card_not_at_a_member():
    """When no plate forms, the members still reference the hoisted card."""
    presentation = _presentation(relational_graph().global_prior_model)

    assert presentation.cards[0].kind == "hoist"
    for index in ("0", "1", "2"):
        assert _pill(presentation, f"{index}/gaussian/centre").badge == "↗ shared"
    assert all(
        link.target_key.startswith("shared across datasets/")
        for link in presentation.links
        if link.kind == "shared"
    )


def test_a_variable_model_hoists_only_what_is_shared():
    presentation = _presentation(variable_graph().global_prior_model)

    assert [pill.text for pill in presentation.cards[0].pills] == ["centre"]
    assert _pill(presentation, "0/centre").badge == "↗ shared"
    assert _pill(presentation, "0/normalization").badge == "independent"


# -- hierarchical: a draw, never a shared marker ----------------------------


def test_a_hierarchical_model_draws_its_centre_and_never_shares_it():
    presentation = _presentation(hierarchical_graph().global_prior_model)

    hyper = presentation.cards[0]
    assert hyper.kind == "hyper"
    assert hyper.title == "HierarchicalFactor0 · GaussianPrior"
    assert [pill.text for pill in hyper.pills] == ["mean", "sigma"]

    centre = _pill(presentation, "0/centre")
    assert centre.state == "drawn"
    assert centre.text == "centre · drawn"
    assert centre.badge == "↗ HierarchicalFactor0"

    # The arrow leaves the hyper card and lands ON the pill.
    draws = [link for link in presentation.links if link.kind == "draw"]
    assert draws == [
        af.model_figure.presentation.Link("HierarchicalFactor0", "0/centre", "draw")
    ]

    # No shared marker anywhere on the figure, and no hoisted card.
    assert all(card.kind != "hoist" for card in presentation.walk())
    assert all(
        link.kind != "shared" for link in presentation.links
    ), "a draw must never be reported as sharing"
    for pill in presentation.pills():
        assert pill.badge != "↗ shared"
        assert not (pill.badge or "").startswith("shared")


def test_the_drawn_badge_is_violet_and_the_shared_badge_is_blue():
    """The two claims are opposites, so they never share a colour."""
    drawn = _pill(_presentation(hierarchical_graph().global_prior_model), "0/centre")
    shared = _pill(_presentation(shared_graph().global_prior_model), "0/centre")

    assert _badge_colours(drawn.badge, drawn.state) == (
        PALETTE["violet_fill"],
        PALETTE["violet"],
    )
    assert _badge_colours(shared.badge, shared.state) == (
        PALETTE["blue_fill"],
        PALETTE["blue"],
    )


def test_the_draw_arrow_ends_on_the_pill_inside_the_plate():
    layout = af.ModelPlotter(hierarchical_graph().global_prior_model).layout(width=14.0)

    route = next(link for link in layout.links if link.kind == "draw")
    pill = next(pill for pill in layout.pills() if pill.key == "0/centre")
    end_x, end_y = route.points[-1]

    assert end_x == round(pill.x, 4)
    assert pill.y <= end_y <= pill.y + pill.height


# -- observed data is not a fixed constant ----------------------------------


def test_observed_data_has_its_own_encoding():
    presentation = _presentation(shared_graph().global_prior_model)

    data = _pill(presentation, "0/data")
    assert data.state == "observed"
    assert data.text == "data · observed"
    assert _PILL_STYLE["observed"] != _PILL_STYLE["fixed"]
    assert "observed data" in presentation.legend


# -- the footer separates hyper-parameters from per-dataset parameters ------


def test_the_hierarchical_footer_separates_the_hyper_parameters():
    presentation = _presentation(hierarchical_graph().global_prior_model)

    assert presentation.footer.startswith(
        "2 hyper-parameters  ·  0 shared across datasets  ·  "
        "3 per dataset × 3 datasets  ·  11 unique sampled scalars"
    )


def test_the_shared_footer_says_nothing_is_per_dataset():
    presentation = _presentation(shared_graph().global_prior_model)

    assert presentation.footer.startswith(
        "3 shared across datasets  ·  0 per dataset × 3 datasets  ·  "
        "3 unique sampled scalars"
    )
    assert "3 observed datasets" in presentation.footer


def test_the_legend_names_only_what_is_on_the_figure():
    hierarchical = _presentation(hierarchical_graph().global_prior_model).legend
    shared = _presentation(shared_graph().global_prior_model).legend

    assert "drawn from a hyper-prior" in hierarchical
    assert "violet arrow = the distribution it is drawn from" in hierarchical
    assert "blue badge = shared prior" not in hierarchical

    assert "blue badge = shared prior" in shared
    assert "drawn from a hyper-prior" not in shared


# -- every graphical figure renders inside the width budget ----------------


def test_every_graphical_figure_renders_within_the_width_budget():
    for builder in (shared_graph, variable_graph, relational_graph, hierarchical_graph):
        af.ModelObject._ids = itertools.count()
        af.Prior._ids = itertools.count()
        plotter = af.ModelPlotter(builder().global_prior_model)
        layout = plotter.layout(width=14.0)

        assert layout.width <= 14.0
        assert layout.width * layout.style.dpi <= 1400

        figure = plotter.figure(width=14.0, format=None)
        assert figure is not None
        plt.close(figure)


# -- nothing changes for a model that is not a factor graph ----------------


#: ``sha256[:16]`` of ``json.dumps(presentation.to_dict())`` for each lens
#: double, captured from the implementation **before** the graphical pass
#: existed.  The phase-4 rules are all gated on a ``global`` root or on a
#: provenance kind only the graphical pass emits, so every one of these must
#: still hash the same: "non-graphical models are presented byte-identically".
_BEFORE_PHASE_4 = {
    "simple_lens": {
        "names": "f8131c06afad0cab",
        "priors": "a80e3e969b507f8f",
    },
    "mge_pixelized": {
        "names": "fa9b8cc21fe33cf3",
        "priors": "963a978c671cae28",
    },
    "group_scale": {
        "names": "48849168782d1068",
        "priors": "ac258bec6dea2d7b",
    },
}

_LENS_DOUBLES = {
    "simple_lens": simple_lens_model,
    "mge_pixelized": mge_model,
    "group_scale": group_model,
}


def test_non_graphical_presentations_are_byte_identical_to_phase_3():
    for name, builder in _LENS_DOUBLES.items():
        af.ModelObject._ids = itertools.count()
        af.Prior._ids = itertools.count()
        model = builder()
        spec = GraphSpec.from_model(model)
        priors = prior_summaries(model)
        for detail in ("names", "priors"):
            blob = json.dumps(
                build_presentation(spec, detail=detail, priors=priors).to_dict()
            )
            digest = hashlib.sha256(blob.encode()).hexdigest()[:16]
            assert digest == _BEFORE_PHASE_4[name][detail], (name, detail)
