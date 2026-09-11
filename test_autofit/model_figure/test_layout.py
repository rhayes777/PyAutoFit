"""
Layer 3a -- the geometry invariants the epic's prototype broke.

Text is never scaled down (so the toy Gaussian and the group-scale model share
every font size), nothing exceeds the width budget, top-level cards wrap rather
than running off the page, declaration order survives placement, and padding
decreases with nesting depth.
"""

import matplotlib

matplotlib.use("Agg")

import pytest

import autofit as af
from autofit.model_figure.layout import Style, build_layout, measure_text
from autofit.model_figure.presentation import build_presentation
from autofit.graph_spec import GraphSpec

from test_autofit.graph_spec.lens_doubles import (
    group_model,
    mge_model,
    simple_lens_model,
)


def _layout(model, width=14.0, **kwargs):
    style = Style(width=width)
    return build_layout(
        build_presentation(GraphSpec.from_model(model), **kwargs), style
    )


MODELS = {
    "gaussian": lambda: af.Model(af.ex.Gaussian),
    "simple_lens": simple_lens_model,
    "mge": mge_model,
    "group": group_model,
}


@pytest.mark.parametrize("name", sorted(MODELS))
def test_no_box_exceeds_the_width_budget(name):
    layout = _layout(MODELS[name]())
    style = layout.style

    assert layout.width <= style.width + 1e-6
    for card in layout.walk():
        assert card.x >= style.margin - 1e-9
        assert card.x + card.width <= style.width - style.gutter + 1e-6
    for pill in layout.pills():
        assert pill.x + pill.width <= style.width - style.gutter + 1e-6


@pytest.mark.parametrize("name", sorted(MODELS))
def test_nothing_is_clipped_vertically(name):
    layout = _layout(MODELS[name]())

    for card in layout.walk():
        assert card.y + card.height <= layout.height + 1e-6
    assert layout.footer_y < layout.height


def test_font_sizes_are_identical_for_the_smallest_and_largest_model():
    """ "Keep text size fixed and collapse content" -- not shrink the figure."""
    styles = [_layout(build()).style.to_dict() for build in MODELS.values()]

    assert all(style == styles[0] for style in styles)
    assert styles[0]["title_size"] == 9.0
    assert styles[0]["pill_size"] == 8.0


def test_top_level_cards_wrap_onto_a_new_row_for_the_group_model():
    layout = _layout(group_model())

    rows = sorted({card.y for card in layout.cards})
    assert len(rows) > 1
    # Reading order across, then down: the first card of every row is flush left.
    assert all(
        any(card.x == layout.style.margin and card.y == row for card in layout.cards)
        for row in rows
    )


def test_wrapping_preserves_declaration_order():
    layout = _layout(group_model())

    positions = [(card.y, card.x) for card in layout.cards]
    assert positions == sorted(positions)
    assert [card.key for card in layout.cards] == ["galaxies", "extra_galaxies"]


def test_children_stack_in_declaration_order():
    layout = _layout(simple_lens_model())
    (galaxies,) = layout.cards
    lens = galaxies.children[0]

    assert [child.key for child in lens.children] == [
        "galaxies/lens/bulge",
        "galaxies/lens/mass",
        "galaxies/lens/shear",
    ]
    ys = [child.y for child in lens.children]
    assert ys == sorted(ys)


def test_pills_flow_in_declaration_order_and_wrap_within_the_card():
    layout = _layout(simple_lens_model())
    bulge = next(card for card in layout.walk() if card.key == "galaxies/lens/bulge")

    positions = [(pill.y, pill.x) for pill in bulge.pills]
    assert positions == sorted(positions)
    for pill in bulge.pills:
        assert pill.x + pill.width <= bulge.x + bulge.width - bulge.pad + 1e-6


def test_padding_decreases_with_nesting_depth():
    style = Style()
    pads = [style.pad(level) for level in range(1, 5)]

    assert pads[0] > pads[1] > pads[2]
    assert pads[-1] >= style.min_pad

    layout = _layout(mge_model())
    by_level = {}
    for card in layout.walk():
        by_level.setdefault(card.level, card.pad)
    assert by_level[1] > by_level[2] > by_level[3]


def test_a_narrow_budget_wraps_rather_than_shrinking_text():
    wide = _layout(group_model(), width=14.0)
    narrow = _layout(group_model(), width=9.0)

    assert narrow.style.pill_size == wide.style.pill_size
    assert narrow.height > wide.height
    for card in narrow.walk():
        assert card.x + card.width <= 9.0 - narrow.style.gutter + 1e-6


def test_text_is_measured_not_estimated():
    """The prototype's estimator was 11 % short on bold headers."""
    bold = measure_text("bulge · Sersic", 9.0, "bold")[0]
    plain = measure_text("bulge · Sersic", 9.0)[0]

    assert bold > plain > 0.0
    # The cache returns the identical object for the identical query.
    assert measure_text("bulge · Sersic", 9.0, "bold") == (
        bold,
        measure_text("bulge · Sersic", 9.0, "bold")[1],
    )


def test_links_are_routed_only_through_the_gutter():
    layout = _layout(mge_model())
    card_right = max(card.x + card.width for card in layout.walk())

    assert layout.links
    for route in layout.links:
        assert all(x >= card_right for x, _ in route.points)
        assert all(x <= layout.width for x, _ in route.points)
        # Orthogonal: every segment changes exactly one coordinate.
        for (x0, y0), (x1, y1) in zip(route.points, route.points[1:]):
            assert x0 == x1 or y0 == y1


def test_link_ends_touch_the_top_level_card_that_holds_the_pill():
    layout = _layout(mge_model())
    tops = {card.key: card for card in layout.cards}

    assert layout.links
    for route in layout.links:
        for key, (x, y) in (
            (route.source_key, route.points[0]),
            (route.target_key, route.points[-1]),
        ):
            owner = next(top for top in tops.values() if key.startswith(f"{top.key}/"))
            assert x >= owner.x + owner.width - 1e-6
            # Touching, not floating: exactly the rightmost card edge at this
            # height, so the segment is attached and still enters no card.
            assert x == max(
                other.x + other.width
                for other in layout.cards
                if other.y <= y <= other.y + other.height
            )


def test_a_large_number_of_links_falls_back_to_labelled_references():
    """ "Use labelled references when links become numerous" (the review)."""
    expanded = build_layout(
        build_presentation(GraphSpec.from_model(mge_model(), collapse=False)),
        Style(),
    )

    assert expanded.links == ()
    assert any(pill.badge and pill.badge.startswith("↗") for pill in expanded.pills())
