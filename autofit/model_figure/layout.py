"""
Layer 3a of the model figure -- measurement and placement.

Everything here is in **inches**, so the drawing layer is a straight
transcription and the figure size falls out of the content rather than being
guessed.  The two rules the epic's prototype broke, and that this module exists
to keep:

* **Measure bottom-up, place top-down.**  Text is measured with
  ``Text.get_window_extent`` on a scratch Agg canvas -- the prototype's width
  *estimator* was 11 % short on bold headers and the figure clipped.
* **Fixed text size; collapse, never shrink.**  Every font size lives on
  :class:`Style` and is the same for the toy Gaussian and the group-scale
  model.  Scalability comes from the presentation layer collapsing content, and
  from top-level cards **wrapping** onto a new row when the width budget is
  exhausted -- never from scaling the figure down.

Declaration order is inviolable: pills flow in the order the presentation
emitted them and children stack in the order they were declared.  Nothing is
reordered to fill space (the prototype's masonry floated ``shear`` above
``mass``).

:meth:`LayoutTree.to_dict` is the determinism artefact: every coordinate is
rounded to 4 dp, so ``json.dumps`` of it is byte-stable for a given model.

matplotlib is imported **inside** :func:`measure_text`, never at module import,
so ``import autofit`` stays drawing-free.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

__all__ = [
    "Style",
    "PillBox",
    "CardBox",
    "ConstraintBox",
    "LinkRoute",
    "LayoutTree",
    "measure_text",
    "build_layout",
]


@dataclass(frozen=True)
class Style:
    """
    Fixed geometry and typography.

    Font sizes never vary with model size.  ``pad`` *decreases* per nesting
    level (review: "reduce padding progressively through nested containers"),
    and ``gutter`` inches of the right-hand edge are reserved for cross links
    and drawn on by nothing else.
    """

    width: float = 14.0
    dpi: int = 100

    title_size: float = 9.0
    pill_size: float = 8.0
    badge_size: float = 7.0
    note_size: float = 7.0
    footer_size: float = 7.5

    margin: float = 0.2
    gutter: float = 0.9

    max_links: int = 12

    base_pad: float = 0.17
    pad_shrink: float = 0.7
    min_pad: float = 0.055

    pill_pad_x: float = 0.075
    pill_pad_y: float = 0.05
    pill_gap: float = 0.07
    inner_gap: float = 0.06
    card_gap: float = 0.13
    title_gap: float = 0.07
    section_gap: float = 0.09
    legend_gap: float = 0.22

    def pad(self, level: int) -> float:
        """Padding of a card at nesting ``level`` (1 = top level)."""
        return round(
            max(self.min_pad, self.base_pad * self.pad_shrink ** (level - 1)), 4
        )

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "dpi": self.dpi,
            "title_size": self.title_size,
            "pill_size": self.pill_size,
            "badge_size": self.badge_size,
            "note_size": self.note_size,
            "footer_size": self.footer_size,
            "gutter": self.gutter,
            "max_links": self.max_links,
        }


# ----------------------------------------------------------------------------
# text measurement
# ----------------------------------------------------------------------------

_CACHE: Dict[Tuple[str, float, str], Tuple[float, float]] = {}
_FIGURE = None


def measure_text(text: str, fontsize: float, weight: str = "normal"):
    """
    The ``(width, height)`` of ``text`` in **inches**, measured, never estimated.

    A scratch ``FigureCanvasAgg`` is used rather than the live figure so the
    measurement is backend independent and available before any figure exists.
    Results are cached per ``(text, size, weight)``.
    """
    key = (text, round(float(fontsize), 3), weight)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    global _FIGURE
    if _FIGURE is None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        _FIGURE = Figure(figsize=(1.0, 1.0), dpi=100)
        FigureCanvasAgg(_FIGURE)

    renderer = _FIGURE.canvas.get_renderer()
    artist = _FIGURE.text(0.0, 0.0, text, fontsize=fontsize, fontweight=weight)
    extent = artist.get_window_extent(renderer)
    artist.remove()
    size = (
        round(extent.width / _FIGURE.dpi, 4),
        round(extent.height / _FIGURE.dpi, 4),
    )
    _CACHE[key] = size
    return size


def _line_height(fontsize: float) -> float:
    """A stable line height -- ``get_window_extent`` of a glyph-free string is 0."""
    return round(measure_text("Ag", fontsize)[1] * 1.35, 4)


# ----------------------------------------------------------------------------
# boxes
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class PillBox:
    key: str
    text: str
    state: str
    dim2d: bool
    badge: Optional[str]
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    text_offset: float = 0.0
    tag_offset: Optional[float] = None
    badge_offset: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "text": self.text,
            "state": self.state,
            "dim2d": self.dim2d,
            "badge": self.badge,
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "width": round(self.width, 4),
            "height": round(self.height, 4),
        }


@dataclass(frozen=True)
class ConstraintBox:
    text: str
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "width": round(self.width, 4),
            "height": round(self.height, 4),
        }


@dataclass(frozen=True)
class CardBox:
    key: str
    kind: str
    title: str
    subtitle: Optional[str] = None
    badge: Optional[str] = None
    note: Optional[str] = None
    level: int = 1
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    title_y: float = 0.0
    subtitle_y: Optional[float] = None
    note_y: Optional[float] = None
    pad: float = 0.0
    pills: Tuple[PillBox, ...] = ()
    children: Tuple["CardBox", ...] = ()
    constraints: Tuple[ConstraintBox, ...] = ()

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "kind": self.kind,
            "title": self.title,
            "subtitle": self.subtitle,
            "badge": self.badge,
            "note": self.note,
            "level": self.level,
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "width": round(self.width, 4),
            "height": round(self.height, 4),
            "pad": round(self.pad, 4),
            "pills": [pill.to_dict() for pill in self.pills],
            "children": [child.to_dict() for child in self.children],
            "constraints": [box.to_dict() for box in self.constraints],
        }


@dataclass(frozen=True)
class LinkRoute:
    source_key: str
    target_key: str
    kind: str
    points: Tuple[Tuple[float, float], ...] = ()

    def to_dict(self) -> dict:
        return {
            "source_key": self.source_key,
            "target_key": self.target_key,
            "kind": self.kind,
            "points": [[round(x, 4), round(y, 4)] for x, y in self.points],
        }


@dataclass(frozen=True)
class LayoutTree:
    """Every box of the figure, in inches, with the origin at the top left."""

    width: float
    height: float
    cards: Tuple[CardBox, ...] = ()
    links: Tuple[LinkRoute, ...] = ()
    legend: str = ""
    footer: str = ""
    legend_y: float = 0.0
    footer_y: float = 0.0
    style: Style = field(default_factory=Style)

    def walk(self):
        def _walk(card):
            yield card
            for child in card.children:
                yield from _walk(child)

        for card in self.cards:
            yield from _walk(card)

    def pills(self):
        return [pill for card in self.walk() for pill in card.pills]

    def to_dict(self) -> dict:
        return {
            "width": round(self.width, 4),
            "height": round(self.height, 4),
            "cards": [card.to_dict() for card in self.cards],
            "links": [link.to_dict() for link in self.links],
            "legend": self.legend,
            "footer": self.footer,
            "legend_y": round(self.legend_y, 4),
            "footer_y": round(self.footer_y, 4),
            "style": self.style.to_dict(),
        }


# ----------------------------------------------------------------------------
# measurement
# ----------------------------------------------------------------------------

#: The tag drawn on a tuple pill so a pill count reconciles with the footer.
TAG_2D = "2D"


def _measure_pill(pill, style: Style) -> PillBox:
    text_w, _ = measure_text(pill.text, style.pill_size)
    height = round(_line_height(style.pill_size) + 2 * style.pill_pad_y, 4)

    x = style.pill_pad_x
    text_offset = x
    x += text_w

    tag_offset = None
    if pill.dim2d:
        x += style.inner_gap
        tag_offset = x
        x += measure_text(TAG_2D, style.badge_size)[0] + 2 * 0.03

    badge_offset = None
    if pill.badge:
        x += style.inner_gap
        badge_offset = x
        x += measure_text(pill.badge, style.badge_size)[0] + 2 * 0.04

    return PillBox(
        key=pill.key,
        text=pill.text,
        state=pill.state,
        dim2d=pill.dim2d,
        badge=pill.badge,
        width=round(x + style.pill_pad_x, 4),
        height=height,
        text_offset=round(text_offset, 4),
        tag_offset=None if tag_offset is None else round(tag_offset, 4),
        badge_offset=None if badge_offset is None else round(badge_offset, 4),
    )


def _wrap(pills: List[PillBox], available: float, style: Style):
    """
    Flow ``pills`` into rows no wider than ``available``.

    Greedy and order preserving: a pill never jumps a row to fill a gap.
    """
    rows: List[List[PillBox]] = []
    current: List[PillBox] = []
    used = 0.0
    for pill in pills:
        step = pill.width if not current else pill.width + style.pill_gap
        if current and used + step > available + 1e-9:
            rows.append(current)
            current = [pill]
            used = pill.width
            continue
        current.append(pill)
        used += step
    if current:
        rows.append(current)
    return rows


def _row_width(row, style: Style) -> float:
    return sum(pill.width for pill in row) + style.pill_gap * (len(row) - 1)


def _measure_card(
    card, level: int, available: float, style: Style, attached
) -> CardBox:
    """
    Size one card and everything inside it.

    ``available`` is the content width this card may occupy.  Children are
    measured first (bottom-up), the pills are wrapped against the widest of the
    title / children / their own natural run, and the card takes the maximum.
    """
    pad = style.pad(level)
    inner = max(available - 2 * pad, 0.6)

    children = [
        _measure_card(child, level + 1, inner, style, attached)
        for child in card.children
    ]
    widest_child = max((child.width for child in children), default=0.0)

    title_w = measure_text(card.title, style.title_size, "bold")[0]
    if card.badge:
        title_w += style.inner_gap + measure_text(card.badge, style.badge_size)[0] + 0.1
    subtitle_w = (
        measure_text(card.subtitle, style.badge_size)[0] if card.subtitle else 0.0
    )
    note_w = measure_text(card.note, style.note_size)[0] if card.note else 0.0

    pills = [_measure_pill(pill, style) for pill in card.pills]
    natural = _row_width(pills, style) if pills else 0.0
    target = min(inner, max(natural, widest_child, title_w))
    rows = _wrap(pills, max(target, 0.6), style) if pills else []

    constraints = []
    for constraint in attached.get(card.key, ()):
        width, _ = measure_text(constraint.text, style.badge_size)
        constraints.append(
            ConstraintBox(
                text=constraint.text,
                width=round(width + 0.16, 4),
                height=round(_line_height(style.badge_size) + 0.08, 4),
            )
        )

    content = max(
        [title_w, subtitle_w, note_w, widest_child]
        + [_row_width(row, style) for row in rows]
        + [box.width for box in constraints]
        + [0.6]
    )

    # -- place, top down --------------------------------------------------
    y = pad
    title_h = _line_height(style.title_size)
    title_y = y
    y += title_h

    subtitle_y = None
    if card.subtitle:
        y += style.title_gap * 0.4
        subtitle_y = y
        y += _line_height(style.badge_size)

    if rows:
        y += style.title_gap
        placed: List[PillBox] = []
        for row in rows:
            x = pad
            for pill in row:
                placed.append(
                    PillBox(
                        key=pill.key,
                        text=pill.text,
                        state=pill.state,
                        dim2d=pill.dim2d,
                        badge=pill.badge,
                        x=round(x, 4),
                        y=round(y, 4),
                        width=pill.width,
                        height=pill.height,
                        text_offset=pill.text_offset,
                        tag_offset=pill.tag_offset,
                        badge_offset=pill.badge_offset,
                    )
                )
                x += pill.width + style.pill_gap
            y += row[0].height + style.pill_gap
        y -= style.pill_gap
        pills = placed
    else:
        pills = []

    if children:
        y += style.section_gap
        placed_children = []
        for child in children:
            placed_children.append(_translate(child, pad, y))
            y += child.height + style.card_gap
        y -= style.card_gap
        children = placed_children

    if constraints:
        y += style.section_gap
        placed_constraints = []
        for box in constraints:
            placed_constraints.append(
                ConstraintBox(
                    box.text, round(pad, 4), round(y, 4), box.width, box.height
                )
            )
            y += box.height + style.inner_gap
        y -= style.inner_gap
        constraints = placed_constraints

    note_y = None
    if card.note:
        y += style.inner_gap
        note_y = y
        y += _line_height(style.note_size)

    return CardBox(
        key=card.key,
        kind=card.kind,
        title=card.title,
        subtitle=card.subtitle,
        badge=card.badge,
        note=card.note,
        level=level,
        width=round(content + 2 * pad, 4),
        height=round(y + pad, 4),
        title_y=round(title_y, 4),
        subtitle_y=None if subtitle_y is None else round(subtitle_y, 4),
        note_y=None if note_y is None else round(note_y, 4),
        pad=pad,
        pills=tuple(pills),
        children=tuple(children),
        constraints=tuple(constraints),
    )


def _translate(card: CardBox, dx: float, dy: float) -> CardBox:
    """Shift a measured card (and its contents) into its parent's frame."""
    return CardBox(
        key=card.key,
        kind=card.kind,
        title=card.title,
        subtitle=card.subtitle,
        badge=card.badge,
        note=card.note,
        level=card.level,
        x=round(card.x + dx, 4),
        y=round(card.y + dy, 4),
        width=card.width,
        height=card.height,
        title_y=round(card.title_y + dy, 4),
        subtitle_y=None if card.subtitle_y is None else round(card.subtitle_y + dy, 4),
        note_y=None if card.note_y is None else round(card.note_y + dy, 4),
        pad=card.pad,
        pills=tuple(
            PillBox(
                key=pill.key,
                text=pill.text,
                state=pill.state,
                dim2d=pill.dim2d,
                badge=pill.badge,
                x=round(pill.x + dx, 4),
                y=round(pill.y + dy, 4),
                width=pill.width,
                height=pill.height,
                text_offset=pill.text_offset,
                tag_offset=pill.tag_offset,
                badge_offset=pill.badge_offset,
            )
            for pill in card.pills
        ),
        children=tuple(_translate(child, dx, dy) for child in card.children),
        constraints=tuple(
            ConstraintBox(
                box.text,
                round(box.x + dx, 4),
                round(box.y + dy, 4),
                box.width,
                box.height,
            )
            for box in card.constraints
        ),
    )


# ----------------------------------------------------------------------------
# the figure
# ----------------------------------------------------------------------------


def _pill_index(cards) -> Dict[str, Tuple[PillBox, CardBox]]:
    """``pill key -> (the pill, the TOP-LEVEL card it lives in)``.

    The top-level card, not the immediate parent: a link leaves the figure's
    content at the outermost frame's edge, because anything closer would have to
    cross that frame's boundary to reach the gutter.
    """
    index: Dict[str, Tuple[PillBox, CardBox]] = {}

    def _walk(card, top):
        for pill in card.pills:
            index.setdefault(pill.key, (pill, top))
        for child in card.children:
            _walk(child, top)

    for card in cards:
        _walk(card, card)
    return index


def _route(links, cards, content_right: float, style: Style):
    """
    Route every cross link through the right-hand gutter.

    The gutter is reserved: nothing else is drawn in it, so a link never runs
    alongside a card boundary (review nit: "A_05's blue connector grazes the
    plate boundary; give it a dedicated gutter").  Each link gets its own lane,
    and every segment is orthogonal.
    """
    if len(links) > style.max_links:
        # "Use labelled references when links become numerous" -- past a dozen
        # routes the gutter is noise, and every referring pill already carries
        # its owner's name.
        return ()
    index = _pill_index(cards)
    routes: List[LinkRoute] = []
    lanes = max(1, int(style.gutter / 0.17))
    for number, link in enumerate(links):
        source = index.get(link.source_key)
        target = index.get(link.target_key)
        if source is None or target is None:
            continue
        sy = round(source[0].y + source[0].height / 2, 4)
        ty = round(target[0].y + target[0].height / 2, 4)
        if abs(sy - ty) < source[0].height:
            # Same visual row: the route would collapse to a meaningless stub in
            # the gutter, and the pill's own badge already names its owner.
            continue
        lane = round(content_right + 0.12 + (number % lanes) * 0.17, 4)
        # Each end touches the right edge of the top-level card that holds its
        # pill, so the bracket is visibly attached to the frame it belongs to --
        # clamped rightwards past anything sharing that wrap row, so the segment
        # still never enters a card.
        start = round(_exit_edge(source[1], cards, sy), 4)
        end = round(_exit_edge(target[1], cards, ty), 4)
        routes.append(
            LinkRoute(
                source_key=link.source_key,
                target_key=link.target_key,
                kind=link.kind,
                points=((start, sy), (lane, sy), (lane, ty), (end, ty)),
            )
        )
    return tuple(routes)


def _exit_edge(card: CardBox, cards, y: float) -> float:
    """
    Where a link leaves ``card``: its right edge, unless a sibling top-level card
    sits further right at the same height, in which case past that one too.
    """
    edge = card.x + card.width
    for other in cards:
        if other is card:
            continue
        if other.y <= y <= other.y + other.height and other.x + other.width > edge:
            edge = other.x + other.width
    return edge


def build_layout(presentation, style: Optional[Style] = None) -> LayoutTree:
    """
    Measure and place a :class:`~autofit.model_figure.presentation.Presentation`.

    Top-level cards run left to right in declaration order and wrap onto a new
    row when the width budget is exhausted -- reading order is across, then
    down, and no card is ever moved to fill a gap.
    """
    style = style or Style()
    budget = max(style.width - 2 * style.margin - style.gutter, 2.0)

    attached: Dict[str, List] = {}
    for constraint in presentation.constraints:
        attached.setdefault(constraint.key, []).append(constraint)

    measured = [
        _measure_card(card, 1, budget, style, attached) for card in presentation.cards
    ]

    placed: List[CardBox] = []
    x = style.margin
    y = style.margin
    row_height = 0.0
    for card in measured:
        if (
            placed
            and x > style.margin
            and x + card.width > style.margin + budget + 1e-9
        ):
            x = style.margin
            y += row_height + style.card_gap
            row_height = 0.0
        placed.append(_translate(card, x, y))
        x += card.width + style.card_gap
        row_height = max(row_height, card.height)
    content_bottom = y + row_height
    content_right = max((card.x + card.width for card in placed), default=style.margin)

    links = _route(presentation.links, placed, content_right, style)

    bottom = content_bottom + style.legend_gap
    legend_y = bottom
    bottom += _line_height(style.footer_size) + 0.04
    footer_y = bottom
    bottom += _line_height(style.footer_size) + style.margin

    # The legend and footer are content too: a figure sized only to its cards
    # clips them (the epic's "no clipped text").
    text_right = style.margin + max(
        (
            measure_text(presentation.legend, style.footer_size)[0]
            if presentation.legend
            else 0.0
        ),
        (
            measure_text(presentation.footer, style.footer_size)[0]
            if presentation.footer
            else 0.0
        ),
    )
    width = round(max(content_right + style.gutter, text_right) + style.margin, 4)
    return LayoutTree(
        width=width,
        height=round(bottom, 4),
        cards=tuple(placed),
        links=links,
        legend=presentation.legend,
        footer=presentation.footer,
        legend_y=round(legend_y, 4),
        footer_y=round(footer_y, 4),
        style=style,
    )
