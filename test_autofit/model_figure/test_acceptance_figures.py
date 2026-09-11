"""
The epic's acceptance cases, asserted on the figure rather than on the spec.

(a) the simple lens, (b) the MGE 2x30 with a pixelized source and (e) the group
model come from the structural doubles in ``test_autofit/graph_spec/lens_doubles.py``
(autofit may never import autogalaxy or autolens).  The toy Gaussian and a
shared / relation / assertion composite complete the set.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import autofit as af
from autofit.graph_spec import GraphSpec

from test_autofit.graph_spec.lens_doubles import (
    group_model,
    mge_model,
    simple_lens_model,
)


def _plotter(model, **kwargs):
    return af.ModelPlotter(model, **kwargs)


def _boxes(presentation):
    """Component *boxes* -- models and plates, never collection frames."""
    return [card for card in presentation.walk() if card.kind in ("model", "plate")]


def _pill(presentation, key):
    for pill in presentation.pills():
        if pill.key == key:
            return pill
    raise AssertionError(f"no pill at {key!r}")


# ------------------------------------------------------------------ (a)


def test_a_simple_lens_draws_six_component_cards_in_model_info_order():
    model = simple_lens_model()
    presentation = _plotter(model).presentation()

    boxes = _boxes(presentation)
    assert [box.title for box in boxes] == [
        "lens · Galaxy",
        "bulge · Sersic",
        "mass · Isothermal",
        "shear · ExternalShear",
        "source · Galaxy",
        "bulge · Sersic",
    ]
    # The same six the spec counts, in the same order.
    spec = GraphSpec.from_model(model)
    assert len(boxes) == len(
        [node for node in spec.components() if node.kind == "model"]
    )

    figure = _plotter(model).figure(format=None)
    assert figure is not None
    plt.close(figure)


def test_a_simple_lens_renders_within_the_width_budget():
    layout = _plotter(simple_lens_model()).layout(width=14.0)

    assert layout.width <= 14.0
    assert layout.width * layout.style.dpi <= 1400


# ------------------------------------------------------------------ (b)


def test_b_mge_draws_eleven_components_and_two_thirty_member_plates():
    model = mge_model()
    presentation = _plotter(model).presentation()

    assert len(_boxes(presentation)) == 11

    plates = [card for card in presentation.walk() if card.kind == "plate"]
    assert [plate.badge for plate in plates] == ["30 components", "30 components"]
    assert [plate.subtitle for plate in plates] == ["0 - 29", "30 - 59"]

    assert "16 unique sampled scalars" in presentation.footer

    owner = _pill(presentation, "galaxies/lens/bulge/profile_list/0/centre")
    assert owner.badge == "shared across group"
    assert owner.dim2d is True


def test_b_mge_shows_the_unset_areas_factor_in_the_missing_state():
    presentation = _plotter(mge_model()).presentation()
    pill = _pill(presentation, "galaxies/source/pixelization/image_mesh/areas_factor")

    assert pill.state == "missing"
    assert "missing configuration" in presentation.legend


def test_b_mge_stays_inside_the_budget_expanded_as_well_as_collapsed():
    plotter = _plotter(mge_model())

    collapsed = plotter.layout(width=14.0)
    expanded = plotter.layout(width=14.0, collapse=False)

    assert collapsed.width <= 14.0
    assert expanded.width <= 14.0
    # Collapse, never shrink: the expanded figure is taller, not smaller-typed.
    assert expanded.height > collapsed.height
    assert expanded.style.pill_size == collapsed.style.pill_size


# ------------------------------------------------------------------ (e)


def test_e_group_scale_shows_every_per_galaxy_fixed_centre():
    """
    Their absence from the prototype read as absence from the model (review,
    "Show fixed parameters by default").
    """
    presentation = _plotter(group_model()).presentation()
    pill = _pill(presentation, "extra_galaxies/0/mass/centre")

    assert pill.state == "fixed-varies"
    assert pill.text == "centre · fixed, varies by member"
    assert pill.dim2d is True
    assert "77 fixed leaf slots" in presentation.footer


def test_e_group_scale_collapses_eight_galaxies_into_one_plate():
    presentation = _plotter(group_model()).presentation()
    plates = [card for card in presentation.walk() if card.kind == "plate"]

    assert len(plates) == 1
    assert plates[0].badge == "8 components"


# ------------------------------------------------------------------ toy


def test_the_toy_gaussian_renders():
    presentation = _plotter(af.Model(af.ex.Gaussian)).presentation()

    assert [pill.text for pill in presentation.pills()] == [
        "centre",
        "normalization",
        "sigma",
    ]
    figure = _plotter(af.Model(af.ex.Gaussian)).figure(format=None)
    assert figure is not None
    plt.close(figure)


def test_a_collection_of_gaussians_collapses_into_a_plate():
    model = af.Collection(a=af.Model(af.ex.Gaussian), b=af.Model(af.ex.Gaussian))
    presentation = _plotter(model).presentation()

    (plate,) = [card for card in presentation.walk() if card.kind == "plate"]
    assert plate.badge == "2 components"


# ------------------------------------------------------------------ composite


def test_shared_relation_and_assertion_are_all_legible_at_once():
    first = af.Model(af.ex.Gaussian)
    second = af.Model(af.ex.Gaussian)
    second.centre = first.centre
    second.sigma = first.sigma * 2.0
    model = af.Collection(a=first, b=second)
    model.add_assertion(first.sigma > 5.0)

    presentation = _plotter(model).presentation()

    # sharing: a badge on the owner, a labelled reference and a link
    assert _pill(presentation, "a/centre").badge == "shared ×2"
    assert _pill(presentation, "b/centre").badge == "↗ a.centre"
    assert [
        (link.source_key, link.target_key, link.kind) for link in presentation.links
    ] == [
        ("b/centre", "a/centre", "shared"),
        ("b/sigma", "a/sigma", "relation"),
    ]

    # relation: the defining expression is the pill
    relation = _pill(presentation, "b/sigma")
    assert relation.state == "relation"
    assert relation.text == "sigma = a.sigma * 2.0"

    # assertion: one compact constraint label carrying both operands
    (constraint,) = presentation.constraints
    assert constraint.text == "assert a.sigma > 5.0"

    # and all three are in one legend line
    assert "relation (expression shown)" in presentation.legend
    assert "blue badge = shared prior" in presentation.legend
    assert "dashed orange = constraint (assertion)" in presentation.legend

    figure = _plotter(model).figure(format=None)
    assert figure is not None
    plt.close(figure)
