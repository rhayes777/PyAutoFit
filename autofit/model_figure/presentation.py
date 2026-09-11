"""
Layer 2 of the model figure -- the presentation transformation.

This module turns a :class:`~autofit.graph_spec.GraphSpec` (layer 1, the
semantic tree) into a :class:`Presentation` (cards, pills, links, constraints,
legend, footer).  It decides **what is drawn and what it says**; it decides
nothing about **where** anything goes and it imports no drawing library --
``layout.py`` measures and places, ``render.py`` draws.

The vocabulary
--------------

The encoding is the epic's v2 vocabulary as amended by the independent review
(``PyAutoMind/draft/feature/autofit/model_figures_epic.md``).  Its two load
bearing ideas:

* **Sharing is a property, not a state.**  A shared prior is still sampled, so
  it keeps its ``free`` pill and gains a blue *badge*.  The first occurrence in
  visual order owns the parameter (``shared across group`` inside a plate,
  ``shared x k`` otherwise); every later occurrence carries ``-> owner.path``
  and a :class:`Link`.
* **Repetition is not sharing.**  A plate badge reads ``30 components``, never
  ``x30``; a constant that differs between plate members reads
  ``fixed, varies by member``; a free prior repeated independently across a
  plate's members is badged ``independent``.

Plate notation for a factor graph
---------------------------------

When the spec's root is a ``GlobalPriorModel`` (``kind == "global"``) the same
vocabulary is arranged as **plate notation**, and three further claims are kept
apart:

* **a draw is not sharing.**  A ``hierarchical-draw`` row is a violet
  ``centre . drawn`` pill badged with the hyper node it came from, and a violet
  **arrow** lands on that pill.  ``->  shared`` says *the same number in every
  dataset*; the draw arrow says *a different number in every dataset, from a
  common population*.  They never share a colour or a marker.
* **what is shared is drawn once, above the plate.**  The priors every dataset
  carries move into a hoisted ``shared across datasets`` card
  (``kind="hoist"``) and the plate's own pill becomes a blue reference to it.
* **observed data is not a fixed constant.**  An ``observed`` row is a green
  ``data . observed`` pill, never the grey fixed one.

The plate reads ``3 datasets`` / ``dataset 0 - 2``, the hyper nodes are
``kind="hyper"`` cards titled ``HierarchicalFactor0 . GaussianPrior``, and the
footer counts hyper-parameters separately from the per-dataset parameters.
Nothing above applies to a non-graphical model, whose presentation is unchanged.

Everything a card or pill stands for is keyed by its
:attr:`~autofit.graph_spec.GraphSpec.path_index` key (the ``"/"``-joined path),
so a static PNG stays navigable against ``model.info``.

Known limitation carried from layer 1
-------------------------------------

A :class:`~autofit.graph_spec.ParamRow` records ``prior_cls_name`` but not the
prior's *parameters*, so ``detail="priors"`` cannot be served from the spec
alone.  :func:`prior_summaries` builds a ``{prior id: "U(0, 100)"}`` map from the
model itself and :func:`build_presentation` accepts it as ``priors=``;
``ModelPlotter`` supplies it.  Without it ``detail="priors"`` degrades to the
prior's class initial rather than inventing numbers.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Pill",
    "Card",
    "Constraint",
    "Link",
    "Presentation",
    "build_presentation",
    "prior_summaries",
    "compact_prior",
]

#: Pill states.  ``observed`` and ``drawn`` come from the graphical pass' own
#: provenance kinds, and appear only on a factor graph's figure.
STATES = (
    "free",
    "fixed",
    "fixed-varies",
    "relation",
    "solved",
    "missing",
    "observed",
    "drawn",
    "folded",
)

#: ``Prior`` class name -> the compact symbol the figure prints.
_PRIOR_SYMBOLS = {
    "UniformPrior": "U",
    "GaussianPrior": "N",
    "LogUniformPrior": "LogU",
    "LogGaussianPrior": "LogN",
    "TruncatedGaussianPrior": "TN",
}

#: The parameters printed for each compact symbol, in order.
_PRIOR_FIELDS = {
    "U": ("lower_limit", "upper_limit"),
    "N": ("mean", "sigma"),
    "LogU": ("lower_limit", "upper_limit"),
    "LogN": ("mean", "sigma"),
    "TN": ("mean", "sigma", "lower_limit", "upper_limit"),
}


# ----------------------------------------------------------------------------
# formatting helpers
# ----------------------------------------------------------------------------


def _value(value) -> str:
    """
    A value written as Python writes it -- ``1.0`` stays ``1.0``.

    Used where the value *is* the statement (the ``redshift = 0.5`` subtitle):
    a redshift printed as ``1`` reads like an index, not a redshift.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    return repr(value)


def _plural(count: int, singular: str, plural: Optional[str] = None) -> str:
    """``1 plate`` / ``2 plates`` / ``0 shared priors``."""
    return f"{count} {singular if count == 1 else (plural or singular + 's')}"


def _number(value) -> str:
    """A compact, locale-free rendering of a number (``100.0`` -> ``100``)."""
    if value is None:
        return "?"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:g}"
    return str(value)


def compact_prior(prior) -> str:
    """
    ``U(0, 100)`` / ``N(0.5, 0.1)`` / ``LogU(1e-06, 1e+06)`` / ``TN(...)``.

    A presentation-layer mapping of the strings ``model.info`` already prints
    (``Prior.parameter_string``) -- never a second source of truth for the
    numbers themselves.
    """
    name = type(prior).__name__
    symbol = _PRIOR_SYMBOLS.get(name)
    if symbol is None:
        return name.replace("Prior", "")
    values = [
        _number(getattr(prior, field_name, None))
        for field_name in _PRIOR_FIELDS[symbol]
    ]
    return f"{symbol}({', '.join(values)})"


def prior_summaries(model) -> Dict[int, str]:
    """
    ``{prior id: compact summary}`` for every prior in ``model``.

    Layer 1 does not carry prior parameters (see the module docstring), so this
    reads them from the model and hands them to :func:`build_presentation`.
    """
    summaries: Dict[int, str] = {}
    for prior in getattr(model, "priors", ()):
        summaries[prior.id] = compact_prior(prior)
    return summaries


# ----------------------------------------------------------------------------
# the presentation data model
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Pill:
    """One parameter slot as it will be drawn."""

    text: str
    state: str
    dim2d: bool = False
    badge: Optional[str] = None
    key: str = ""

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "state": self.state,
            "dim2d": self.dim2d,
            "badge": self.badge,
            "key": self.key,
        }


@dataclass(frozen=True)
class Card:
    """One component, collection frame or plate as it will be drawn."""

    title: str
    kind: str
    subtitle: Optional[str] = None
    pills: Tuple[Pill, ...] = ()
    children: Tuple["Card", ...] = ()
    badge: Optional[str] = None
    note: Optional[str] = None
    key: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "kind": self.kind,
            "subtitle": self.subtitle,
            "pills": [pill.to_dict() for pill in self.pills],
            "children": [child.to_dict() for child in self.children],
            "badge": self.badge,
            "note": self.note,
            "key": self.key,
        }


@dataclass(frozen=True)
class Constraint:
    """
    An assertion, drawn as a compact label carrying both operand paths.

    ``key`` is the key of the card it is attached to -- the deepest card
    containing both operands, or the first top-level card when neither operand
    resolves into the tree.  Assertions are never edges (review: "long orange
    routes past several junctions are not acceptable").
    """

    text: str
    left: str
    right: str
    key: str = ""

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "left": self.left,
            "right": self.right,
            "key": self.key,
        }


@dataclass(frozen=True)
class Link:
    """A cross reference drawn through the figure's right-hand gutter."""

    source_key: str
    target_key: str
    kind: str

    def to_dict(self) -> dict:
        return {
            "source_key": self.source_key,
            "target_key": self.target_key,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class Presentation:
    """Everything the layout engine needs, and nothing about pixels."""

    cards: Tuple[Card, ...] = ()
    links: Tuple[Link, ...] = ()
    constraints: Tuple[Constraint, ...] = ()
    legend: str = ""
    footer: str = ""
    hidden_fixed: int = 0

    def to_dict(self) -> dict:
        return {
            "cards": [card.to_dict() for card in self.cards],
            "links": [link.to_dict() for link in self.links],
            "constraints": [constraint.to_dict() for constraint in self.constraints],
            "legend": self.legend,
            "footer": self.footer,
            "hidden_fixed": self.hidden_fixed,
        }

    def walk(self):
        """Every card, depth first, in visual order."""

        def _walk(card):
            yield card
            for child in card.children:
                yield from _walk(child)

        for card in self.cards:
            yield from _walk(card)

    def pills(self):
        """Every pill of every card, in visual order."""
        return [pill for card in self.walk() for pill in card.pills]


# ----------------------------------------------------------------------------
# the transformation
# ----------------------------------------------------------------------------


def _dotted(key: str) -> str:
    return key.replace("/", ".")


def _key(path) -> str:
    return "/".join(path)


class _Context:
    """
    Everything the per-rule functions need to look at the whole spec at once.

    Built once per :func:`build_presentation` call: the member -> representative
    prefix map a collapse leaves behind, the visual order of every row, and the
    plate each node sits in.
    """

    def __init__(self, spec, priors, detail):
        self.spec = spec
        self.priors = priors or {}
        self.detail = detail

        self.plates = [node for node in spec.components() if node.plate is not None]
        #: member path -> the representative (plate) path it collapses onto.
        self.representatives: List[Tuple[tuple, tuple]] = []
        for plate in self.plates:
            for member in plate.plate.member_paths[1:]:
                self.representatives.append((member, plate.path))

        self.order: Dict[str, int] = {}
        self.plate_of: Dict[str, Any] = {}
        self.node_depth: Dict[str, int] = {}
        #: every path a drawn row's prior occupies -> that row's key.  Layer 1
        #: names a relation operand by the slot the *compound prior* holds it in
        #: (``b.sigma.self``), which is a real path but not the one the reader
        #: knows; this maps it back onto the row that is drawn (``a.sigma``).
        self.by_occurrence: Dict[str, str] = {}
        for node, plate, depth in _walk(spec.root):
            self.plate_of[_key(node.path)] = plate
            self.node_depth[_key(node.path)] = depth
            for row in node.rows:
                self.order.setdefault(_key(row.path), len(self.order))
                self._record_occurrences(row)
                for component in row.components:
                    self.order.setdefault(_key(component.path), len(self.order))
                    self._record_occurrences(component)

        #: row key -> the row itself, for the rules that need it back.
        self.row_of: Dict[str, Any] = {}
        for node, _, _ in _walk(spec.root):
            for row in node.rows:
                self.row_of.setdefault(_key(row.path), row)
                for component in row.components:
                    self.row_of.setdefault(_key(component.path), component)

        #: drawn pill key -> the key of the hyper card it is drawn from.  The
        #: draw is looked up on the *representative*, because a plate draws only
        #: its first member (the draw is the same statement for every member).
        self.draws: Dict[str, str] = {}
        for edge in getattr(spec, "draws", ()) or ():
            target = _key(self.representative(edge.target_path))
            self.draws.setdefault(target, _key(edge.source_path))

        self.states: set = set()
        self.badges: set = set()
        #: referring pill key -> owner pill key, for the cross links.
        self.references: Dict[str, str] = {}
        #: drawn pill key -> hyper card key, for the violet draw links.
        self.draw_references: Dict[str, str] = {}

    def _record_occurrences(self, row):
        key = _key(row.path)
        for occurrence in row.occurrences:
            self.by_occurrence.setdefault(_key(occurrence), key)

    # -- collapse-aware path resolution -------------------------------------

    def representative(self, path) -> tuple:
        """
        ``path`` with any collapsed-member prefix replaced by its plate's.

        An occurrence inside member 7 of an 8-member plate is drawn on the
        representative, so that is the path a link must point at.
        """
        for member, plate_path in self.representatives:
            if len(path) >= len(member) and tuple(path[: len(member)]) == member:
                return tuple(plate_path) + tuple(path[len(member) :])
        return tuple(path)

    def rendered(self, path) -> Optional[str]:
        """The key ``path`` is drawn at, or ``None`` if it is not drawn."""
        key = _key(self.representative(path))
        if key in self.order:
            return key
        owner = self.by_occurrence.get(key)
        if owner is not None:
            return owner
        if path and path[-1] == "self":
            return self.rendered(tuple(path)[:-1])
        return None

    def resolve(self, operand: str) -> str:
        """
        An operand name rewritten as the dotted path the figure actually draws.

        ``b.sigma.self * 2.0`` is what layer 1 emits for ``b.sigma = a.sigma *
        2.0``; the reader is looking for ``a.sigma``.
        """
        if not operand:
            return operand
        key = self.rendered(tuple(operand.split(".")))
        return _dotted(key) if key else operand

    def resolve_expression(self, expression: str, operands) -> str:
        """``expression`` with every operand rewritten by :meth:`resolve`."""
        if not expression:
            return expression
        for operand in sorted(operands or (), key=len, reverse=True):
            resolved = self.resolve(operand)
            if resolved != operand:
                expression = expression.replace(operand, resolved)
        return expression


def _walk(node, plate=None, depth=1):
    """``(node, enclosing plate node, depth)`` for every node, in visual order."""
    plate = node if node.plate is not None else plate
    yield node, plate, depth
    for child in node.children:
        yield from _walk(child, plate, depth + 1)


# -- rule: sharing -----------------------------------------------------------


def _owner_key(row, context) -> Optional[str]:
    """
    The key of the *first* occurrence of this row's prior, in visual order.

    That occurrence owns the parameter; every later one refers back to it.
    """
    if not row.shared:
        return None
    keys = []
    for occurrence in row.direct_occurrences:
        key = context.rendered(occurrence)
        if key is not None and key not in keys:
            keys.append(key)
    if not keys:
        return None
    return min(keys, key=lambda key: context.order[key])


def _reference(source: str, owner: str) -> str:
    """
    ``owner`` written relative to ``source`` -- a *labelled reference*.

    The review asks for "labelled references when links become numerous", and
    the full path of a plate member's owner
    (``galaxies.lens.bulge.profile_list.0.centre``) is wider than the card that
    carries it.  The shared prefix is dropped; :attr:`Link.target_key` keeps the
    whole key, so the presentation stays navigable.
    """
    source_parts = source.split("/")
    owner_parts = owner.split("/")
    common = 0
    while (
        common < len(source_parts) - 1
        and common < len(owner_parts) - 1
        and source_parts[common] == owner_parts[common]
    ):
        common += 1
    return ".".join(owner_parts[common:]) or _dotted(owner)


def _shared_descriptor(row, node, plate, context) -> Optional[str]:
    """
    The blue badge a shared row carries, or ``None``.

    ``shared across group`` is used when every occurrence lies inside one
    plate's members (or the prior is one the plate shares in all of them) -- the
    review's separation of repetition from sharing.  Otherwise the owner is
    badged ``shared ×k`` and every other occurrence points back at it.
    """
    owner = _owner_key(row, context)
    if owner is None:
        return None
    key = _key(row.path)
    if owner != key:
        return f"↗ {_reference(key, owner)}"
    if _is_group_shared(row, plate, context):
        return "shared across group"
    return f"shared ×{len(row.direct_occurrences)}"


def _is_group_shared(row, plate, context) -> bool:
    if plate is not None and row.prior_id is not None:
        if row.prior_id in plate.plate.shared_in_all:
            return True
    for candidate in context.plates:
        members = candidate.plate.member_paths
        if len(members) < 2:
            continue
        if all(
            any(
                len(occurrence) >= len(member)
                and tuple(occurrence[: len(member)]) == member
                for member in members
            )
            for occurrence in row.direct_occurrences
        ):
            return True
    return False


def _draw_descriptor(row, context) -> Optional[str]:
    """
    The badge a hierarchically drawn row carries: ``↗ HierarchicalFactor0``.

    It is deliberately *not* blue -- the pill is violet and so is its arrow,
    because "drawn from a common population" and "the same number" are opposite
    claims (review: "replace the shared marker with a hierarchical-draw
    annotation").
    """
    if row.provenance.kind != "hierarchical-draw":
        return None
    key = _key(row.path)
    source = context.draws.get(key)
    if source is None:
        return None
    context.draw_references[key] = source
    return f"↗ {_dotted(source)}"


def _badge(row, node, plate, context) -> Optional[str]:
    """The badge of a single (scalar) row: a draw first, then sharing, then
    independence."""
    drawn = _draw_descriptor(row, context)
    if drawn is not None:
        return drawn
    shared = _shared_descriptor(row, node, plate, context)
    if shared is not None:
        return shared
    if (
        plate is not None
        and plate.plate.count > 1
        and row.sampling == "free"
        and row.provenance.kind != "relation"
        and not row.shared
    ):
        # Distinct prior objects with identical configuration, one per member --
        # repetition, never sharing (review point 2).
        return "independent"
    return None


# -- rule: state -------------------------------------------------------------


def _varies(row, plate) -> bool:
    if plate is None:
        return False
    relative = ".".join(row.path[len(plate.path) :])
    return relative in plate.plate.varies_by_member


def _state(row, plate) -> str:
    kind = row.provenance.kind
    if kind == "relation":
        return "relation"
    if kind == "hierarchical-draw":
        return "drawn"
    if kind == "observed":
        return "observed"
    if row.sampling == "solved":
        return "solved"
    if row.sampling == "missing":
        return "missing"
    if row.sampling == "fixed":
        return "fixed-varies" if _varies(row, plate) else "fixed"
    return "free"


def _text(row, state, context) -> str:
    """The pill's text -- name first, always."""
    name = row.name
    if state == "relation":
        expression = context.resolve_expression(
            row.provenance.expression, row.provenance.operands
        )
        return f"{name} = {expression}"
    if state == "drawn":
        # "drawn", never a sharing marker: a drawn parameter is a *different*
        # number in every dataset, from a common population.
        return f"{name} · drawn"
    if state == "observed":
        return f"{name} · observed"
    if state == "missing":
        return f"{name} · missing"
    if state == "solved":
        # A solved quantity is absent from the model, so there is no prior and
        # no value to summarise -- the annotation *is* the whole text, at every
        # detail level.
        return f"{name} · solved"
    if state == "fixed-varies":
        return f"{name} · fixed, varies by member"
    if context.detail != "priors":
        return name
    if state == "fixed":
        if row.dimensionality == "tuple":
            values = ", ".join(_number(component.value) for component in row.components)
            return f"{name} = ({values})"
        if row.is_instance:
            return f"{name} = {row.prior_cls_name}"
        return f"{name} = {_number(row.value)}"
    summary = context.priors.get(row.prior_id)
    if summary is None and row.components:
        summary = context.priors.get(row.components[0].prior_id)
    if summary is None:
        summary = _PRIOR_SYMBOLS.get(row.prior_cls_name or "", "")
    return f"{name}  {summary}".rstrip()


# -- rule: rows -> pills -----------------------------------------------------


def _leaf_slots(row) -> int:
    return len(row.components) if row.dimensionality == "tuple" else 1


def _pill(row, node, plate, context) -> Pill:
    state = _state(row, plate)
    key = _key(row.path)
    owner = _owner_key(row, context)
    if owner is not None and owner != key:
        context.references[key] = owner
    pill = Pill(
        text=_text(row, state, context),
        state=state,
        dim2d=row.dimensionality == "tuple",
        badge=_badge(row, node, plate, context),
        key=key,
    )
    context.states.add(state)
    if pill.badge:
        context.badges.add(_badge_token(pill))
    return pill


def _badge_token(pill) -> str:
    """
    The legend token a badge contributes.

    A draw badge is ``draw``, never ``↗``: it must not make the legend claim a
    blue sharing badge is on the figure when nothing on it is shared.
    """
    if pill.state == "drawn":
        return "draw"
    if pill.badge.startswith("↗"):
        return "↗"
    return pill.badge


def _tuple_is_uniform(row, node, plate, context) -> bool:
    """
    Whether a tuple's components agree on sampling *and* sharing.

    They must, or the tuple is expanded: a partially fixed or partially shared
    tuple flattened into one pill is a lie (review, "Encoding critique").
    """
    descriptors = {
        (
            _state(component, plate),
            _shared_descriptor(component, node, plate, context) is not None,
            (_shared_descriptor(component, node, plate, context) or "").startswith("↗"),
        )
        for component in row.components
    }
    return len(descriptors) == 1


def _tuple_badge(row, node, plate, context) -> Optional[str]:
    """The merged tuple pill's badge, derived from its (agreeing) components."""
    badges = [_badge(component, node, plate, context) for component in row.components]
    badge = badges[0]
    if badge is None:
        return None
    if badge.startswith("↗"):
        # A component's owner is a slot (``...centre.centre_0``); the merged
        # pill stands for the whole tuple, so it points at the owning tuple.
        component_owner = _owner_key(row.components[0], context)
        if component_owner is None:
            return badge
        owner = component_owner.rsplit("/", 1)[0]
        key = _key(row.path)
        context.references[key] = owner
        return f"↗ {_reference(key, owner)}"
    return badge


def _pills(node, plate, context, show_fixed) -> Tuple[List[Pill], int, Optional[str]]:
    """
    Every drawn pill of one component, the fixed slots hidden, and the redshift
    subtitle when the redshift exception applies.
    """
    pills: List[Pill] = []
    hidden = 0
    redshift = None
    for row in node.rows:
        state = _state(row, plate)
        if (
            node.kind == "model"
            and row.name == "redshift"
            and state == "fixed"
            and row.dimensionality == "scalar"
        ):
            # A presentation convenience, not a different semantic rule: a free
            # redshift keeps the ordinary sampled-parameter pill.
            redshift = f"redshift = {_value(row.value)}"
            continue
        if not show_fixed and state in ("fixed", "fixed-varies"):
            hidden += _leaf_slots(row)
            continue
        if row.dimensionality == "tuple" and row.components:
            if _tuple_is_uniform(row, node, plate, context):
                # One pill with a `2D` tag: the components agree, so merging
                # them hides nothing.
                badge = _tuple_badge(row, node, plate, context)
                pills.append(
                    Pill(
                        text=_text(row, state, context),
                        state=state,
                        dim2d=True,
                        badge=badge,
                        key=_key(row.path),
                    )
                )
                context.states.add(state)
                if badge:
                    context.badges.add("↗" if badge.startswith("↗") else badge)
                continue
            # Mixed sampling or mixed sharing: expand rather than lie.
            for component in row.components:
                pills.append(_pill(component, node, plate, context))
            continue
        pills.append(_pill(row, node, plate, context))
    return pills, hidden, redshift


# -- rule: components -> cards ----------------------------------------------


def _title(node) -> str:
    if node.plate is not None:
        # The member index ("0") is meaningless for a plate; the class is not.
        return node.cls_name
    if node.kind == "collection":
        return node.name
    return f"{node.name} · {node.cls_name}"


def _fold_summary(node) -> str:
    components = 0
    priors = 0
    for child, _, _ in _walk(node):
        components += 1
        for row in child.rows:
            if row.dimensionality == "tuple":
                priors += sum(
                    1 for component in row.components if component.sampling == "free"
                )
            elif row.sampling == "free":
                priors += 1
    return f"… {components} components / {priors} priors"


def _is_dataset_plate(node) -> bool:
    """Whether this plate's members are one ``AnalysisFactor``'s model each."""
    factor = getattr(node, "factor", None)
    return node.plate is not None and factor is not None and factor.kind == "analysis"


def _card(node, plate, depth, context, show_fixed, max_depth) -> Tuple[Card, int]:
    pills, hidden, redshift = _pills(node, plate, context, show_fixed)

    children: List[Card] = []
    for child in node.children:
        child_plate = child if child.plate is not None else plate
        if max_depth is not None and depth + 1 > max_depth:
            pills.append(
                Pill(
                    text=_fold_summary(child),
                    state="folded",
                    key=_key(child.path),
                )
            )
            context.states.add("folded")
            continue
        card, child_hidden = _card(
            child, child_plate, depth + 1, context, show_fixed, max_depth
        )
        children.append(card)
        hidden += child_hidden

    datasets = _is_dataset_plate(node)

    subtitle = None
    if node.plate is not None:
        # A plate over a factor graph's model factors repeats *datasets*, and a
        # reader looking for dataset 2 must be able to find it: "dataset 0 - 2",
        # not a bare index range.
        subtitle = node.plate.representative_key
        if datasets and subtitle:
            subtitle = f"dataset {subtitle}"
        if redshift:
            subtitle = f"{subtitle} · {redshift}" if subtitle else redshift
    elif redshift:
        subtitle = redshift

    badge = None
    note = None
    if not pills and not children and node.plate is None:
        # An empty card is not an error and must say so: the review's
        # "A_06's empty Hilbert card needs an explicit explanation".
        note = "no parameters"
    if node.plate is not None:
        # "30 components", never "x30": a plate counts components, a badge on a
        # parameter counts uses of a parameter (review point 2).
        badge = _plural(node.plate.count, "dataset" if datasets else "component")
        note = node.plate.repeats[0] if node.plate.repeats else None
        context.badges.add("datasets" if datasets else "plate")

    return (
        Card(
            title=_title(node),
            kind="plate" if node.plate is not None else node.kind,
            subtitle=subtitle,
            pills=tuple(pills),
            children=tuple(children),
            badge=badge,
            note=note,
            key=_key(node.path),
        ),
        hidden,
    )


# -- rule: assertions and relations -----------------------------------------


def _operand_key(operand: str, context) -> Optional[str]:
    """The drawn key an assertion / relation operand resolves to, if any."""
    if not operand:
        return None
    path = tuple(operand.split("."))
    for candidate in (path, path[:-1] if path[-1] == "self" else path):
        key = context.rendered(candidate)
        if key is not None:
            return key
    return None


def _deepest_card(keys: Sequence[str], cards: Sequence[Card]) -> str:
    """
    The key of the deepest card containing every key in ``keys``.

    Falls back to the first top-level card, so a constraint whose operands do
    not resolve into the tree is still drawn rather than dropped.
    """
    candidates = []
    for card in _cards_of(cards):
        prefix = card.key
        if all(
            key == prefix or (prefix == "" or key.startswith(f"{prefix}/"))
            for key in keys
        ):
            candidates.append(card.key)
    if not candidates:
        return cards[0].key if cards else ""
    return max(candidates, key=len)


def _cards_of(cards):
    for card in cards:
        yield card
        yield from _cards_of(card.children)


#: Comparison operators, mapped to the operator that says the same thing with
#: the operands the other way round.
_INVERTED_OPERATORS = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}


def _oriented(left: str, op: str, right: str, context):
    """
    The assertion with its **model-side operand first**, when that is unambiguous.

    Layer 1 emits ``5.0 < a.sigma`` for ``add_assertion(a.sigma > 5.0)`` -- true,
    but back to front for a reader looking for ``a.sigma``.  The operands are
    swapped (and the operator inverted) only when exactly one side resolves to a
    drawn parameter and the operator has an inverse; anything else keeps the
    spec's order rather than guessing.
    """
    if op not in _INVERTED_OPERATORS:
        return left, op, right
    left_is_model = _operand_key(left, context) is not None
    right_is_model = _operand_key(right, context) is not None
    if right_is_model and not left_is_model:
        return right, _INVERTED_OPERATORS[op], left
    return left, op, right


def _constraints(spec, cards, context) -> Tuple[Constraint, ...]:
    constraints = []
    for assertion in spec.assertions:
        left, op, right = _oriented(
            context.resolve(assertion.left),
            assertion.op,
            context.resolve(assertion.right),
            context,
        )
        keys = [
            key
            for key in (
                _operand_key(left, context),
                _operand_key(right, context),
            )
            if key is not None
        ]
        constraints.append(
            Constraint(
                # "assert" in the label, a dashed outline in the drawing: a
                # constraint is not a relation, and must not read as one.
                text=f"assert {left} {op} {right}",
                left=left,
                right=right,
                key=_deepest_card(keys, cards) if cards else "",
            )
        )
    return tuple(constraints)


def _links(spec, presentation_keys, context) -> Tuple[Link, ...]:
    links: List[Link] = []
    seen = set()
    drawn = {key for key, _ in presentation_keys}
    for pill_key in sorted(
        context.references, key=lambda key: context.order.get(key, 0)
    ):
        target = context.references[pill_key]
        if pill_key in drawn and target in context.order:
            seen.add((pill_key, target))
            links.append(Link(pill_key, target, "shared"))
    for pill_key in sorted(
        context.draw_references, key=lambda key: context.order.get(key, 0)
    ):
        # The arrow runs *from* the hyper card *to* the drawn pill, and lands on
        # the pill -- not on the dataset frame (review, A_08).  Its target is a
        # card key, which the layout resolves like any other anchor.
        source = context.draw_references[pill_key]
        if pill_key in drawn:
            links.append(Link(source, pill_key, "draw"))
    for relation in spec.relations:
        target = context.rendered(relation.target_path)
        if target is None:
            continue
        for operand in relation.operand_paths:
            source = context.rendered(operand)
            if source is None or source == target:
                continue
            if (target, source) in seen:
                continue
            seen.add((target, source))
            links.append(Link(target, source, "relation"))
    return tuple(links)


# -- rule: hoisting the shared priors of a factor graph ----------------------


#: The key of the synthetic hoisted card.  It cannot collide with a model path:
#: a ``GlobalPriorModel``'s children are list indices.
HOIST_KEY = "shared across datasets"


def _dataset_nodes(spec) -> List[Any]:
    """The root children that stand for an ``AnalysisFactor``'s dataset model."""
    return [
        child
        for child in spec.root.children
        if getattr(child, "factor", None) is not None
        and child.factor.kind == "analysis"
    ]


def _shared_across_datasets(spec, nodes) -> List[int]:
    """
    The prior ids drawn **once** above the plate.

    From the dataset plate's ``shared_in_all`` when the plate formed -- the ids
    every member carries, which is exactly "one prior, N datasets" -- and
    otherwise from the ids every dataset child holds directly.
    """
    plate = next((node for node in nodes if node.plate is not None), None)
    if plate is not None:
        return list(plate.plate.shared_in_all)
    if len(nodes) < 2:
        return []
    per_node = []
    for node in nodes:
        ids = set()
        for current, _, _ in _walk(node):
            for row in current.rows:
                for candidate in (row,) + tuple(row.components):
                    if candidate.prior_id is not None and candidate.shared:
                        ids.add(candidate.prior_id)
        per_node.append(ids)
    common = set.intersection(*per_node)
    return sorted(common)


def _hoist_shared(spec, cards, context) -> Tuple[Tuple[Card, ...], Tuple[Link, ...]]:
    """
    Draw a factor graph's shared priors **once**, above the plate.

    Plate notation's whole point: a parameter outside the plate is one number
    for every dataset.  The owning pill moves to a hoisted
    ``shared across datasets`` card and the plate's own pill becomes a
    reference (``↗ shared``) with a blue link to it, so the figure states the
    sharing in its layout rather than only in a badge.
    """
    if spec.root.kind != "global":
        return cards, ()
    nodes = _dataset_nodes(spec)
    if not nodes:
        return cards, ()
    shared_ids = _shared_across_datasets(spec, nodes)
    if not shared_ids:
        return cards, ()

    drawn_pills = {
        pill.key: pill for card in _cards_of(cards) for pill in card.pills
    }
    hoisted: List[Pill] = []
    links: List[Link] = []
    moved: Dict[str, Pill] = {}
    keys_of: Dict[int, str] = {}
    for key, pill in drawn_pills.items():
        row = context.row_of.get(key)
        if row is None or row.prior_id not in shared_ids:
            continue
        hoist_key = keys_of.get(row.prior_id)
        if hoist_key is None:
            hoist_key = f"{HOIST_KEY}/{row.name}"
            while any(existing.key == hoist_key for existing in hoisted):
                hoist_key = f"{hoist_key}'"
            keys_of[row.prior_id] = hoist_key
            hoisted.append(
                Pill(text=pill.text, state=pill.state, dim2d=pill.dim2d, key=hoist_key)
            )
        # *Every* occurrence points at the hoisted pill, not at whichever member
        # happened to come first: the parameter now lives above the plate.
        moved[key] = Pill(
            text=pill.text,
            state=pill.state,
            dim2d=pill.dim2d,
            badge="↗ shared",
            key=key,
        )
        context.references.pop(key, None)
        links.append(Link(key, hoist_key, "shared"))
    if not hoisted:
        return cards, ()

    context.badges.add("↗")

    def _rewrite(card: Card) -> Card:
        return Card(
            title=card.title,
            kind=card.kind,
            subtitle=card.subtitle,
            pills=tuple(moved.get(pill.key, pill) for pill in card.pills),
            children=tuple(_rewrite(child) for child in card.children),
            badge=card.badge,
            note=card.note,
            key=card.key,
        )

    hoist = Card(
        title=HOIST_KEY,
        kind="hoist",
        subtitle="one value across "
        + _plural(spec.counts.get("datasets", len(nodes)), "dataset"),
        pills=tuple(hoisted),
        key=HOIST_KEY,
    )
    return (hoist,) + tuple(_rewrite(card) for card in cards), tuple(links)


# -- rule: legend and footer -------------------------------------------------

_LEGEND_WORDS = {
    "free": "free prior",
    "fixed": "fixed value",
    "fixed-varies": "fixed, varies by member",
    "relation": "relation (expression shown)",
    "solved": "solved during fitting (dashed)",
    "missing": "missing configuration (red)",
    "observed": "observed data",
    "drawn": "drawn from a hyper-prior",
    "folded": "folded subtree",
}


def _legend(context, constraints=()) -> str:
    parts = [_LEGEND_WORDS[state] for state in STATES if state in context.states]
    if any(badge.startswith("shared") or badge == "↗" for badge in context.badges):
        parts.append("blue badge = shared prior")
    if "draw" in context.badges:
        parts.append("violet arrow = the distribution it is drawn from")
    if "independent" in context.badges:
        parts.append("independent = one prior per member")
    if "plate" in context.badges:
        parts.append("dashed frame = repeated components")
    if "datasets" in context.badges:
        parts.append("dashed frame = one component per dataset")
    if constraints:
        parts.append("dashed orange = constraint (assertion)")
    return "Legend:  " + "  ·  ".join(parts)


def _plate_members(spec) -> int:
    """How many components the plates stand for -- the sum of their counts."""
    return sum(node.plate.count for node in spec.components() if node.plate is not None)


def _graphical_footer_parts(counts) -> List[str]:
    """
    The graphical split of the accounting.

    ``hyper-parameters`` are counted **separately** from the per-dataset
    parameters (review, A_08: "include the two free hyperparameters in the
    accounting, separately from the three parameters per dataset"), and the
    three terms reconcile to the unique sampled total that follows them.
    """
    parts: List[str] = []
    if counts.get("hyper_parameters"):
        parts.append(_plural(counts["hyper_parameters"], "hyper-parameter"))
    parts.append(f"{counts.get('shared_across_datasets', 0)} shared across datasets")
    parts.append(
        f"{counts.get('per_dataset', 0)} per dataset × "
        f"{_plural(counts.get('datasets', 0), 'dataset')}"
    )
    return parts


def _footer(spec, hidden_fixed: int) -> str:
    counts = spec.counts
    graphical = "datasets" in counts
    parts = _graphical_footer_parts(counts) if graphical else []
    parts.append(
        _plural(counts.get("unique_sampled_scalars", 0), "unique sampled scalar")
    )
    parts.append(_plural(counts.get("fixed_leaf_slots", 0), "fixed leaf slot"))
    if not graphical:
        parts.append(
            _plural(counts.get("shared_priors", 0), "shared prior")
            + " (unique variables, not references)"
        )
    if counts.get("observed"):
        parts.append(_plural(counts["observed"], "observed dataset"))
    if counts.get("missing"):
        parts.append(f"{counts['missing']} missing")
    if counts.get("solved"):
        parts.append(
            _plural(
                counts["solved"],
                "parameter solved during fitting",
                "parameters solved during fitting",
            )
        )
    plates = counts.get("plates") or 0
    if plates:
        # What the plates stand for, not how many boxes the tree lost: the
        # latter counts collection frames and muddles the 11-boxes acceptance.
        parts.append(
            f"{_plural(plates, 'plate')} standing for "
            f"{_plural(_plate_members(spec), 'component')}"
        )
    if hidden_fixed:
        parts.append(_plural(hidden_fixed, "fixed parameter") + " hidden")
    else:
        parts.append("totals include every hidden and collapsed element")
    return "  ·  ".join(parts)


# -- the entry point ---------------------------------------------------------


def build_presentation(
    spec,
    detail: str = "names",
    max_depth: Optional[int] = None,
    show_fixed: bool = True,
    priors: Optional[Dict[int, str]] = None,
) -> Presentation:
    """
    Turn a :class:`~autofit.graph_spec.GraphSpec` into a :class:`Presentation`.

    Parameters
    ----------
    spec
        The semantic tree (layer 1).
    detail
        ``"names"`` -- the default map, name-only pills -- or ``"priors"``,
        which adds each pill's compact prior summary or fixed value.
    max_depth
        Cards deeper than this fold into one ``… N components / M priors`` row.
    show_fixed
        Fixed pills are shown by default because they explain model structure.
        Hiding them prints a visible hidden count in the footer.
    priors
        ``{prior id: "U(0, 100)"}``, from :func:`prior_summaries`.  Required for
        ``detail="priors"`` (layer 1 does not carry prior parameters).

    Notes
    -----
    The outer frame is dropped when it adds no branching information -- a root
    ``Collection`` that carries no rows of its own contributes nothing but a box
    around its children, so its children become the top-level cards.  A root
    ``Model`` keeps its card: it owns rows.
    """
    if detail not in ("names", "priors"):
        raise ValueError(f"detail must be 'names' or 'priors', not {detail!r}")

    context = _Context(spec, priors, detail)
    root = spec.root

    hidden_fixed = 0
    cards: List[Card] = []
    if root.kind in ("collection", "global") and not root.rows and root.children:
        for child in root.children:
            plate = child if child.plate is not None else None
            card, hidden = _card(child, plate, 1, context, show_fixed, max_depth)
            cards.append(card)
            hidden_fixed += hidden
    else:
        plate = root if root.plate is not None else None
        card, hidden_fixed = _card(root, plate, 1, context, show_fixed, max_depth)
        cards.append(card)

    cards, hoist_links = _hoist_shared(spec, tuple(cards), context)
    cards = list(cards)
    presentation_keys = [
        (pill.key, pill.badge) for card in _cards_of(cards) for pill in card.pills
    ]
    drawn = {key for key, _ in presentation_keys}
    # A draw link lands on a pill but leaves a *card*, so card keys are anchors
    # too; every other link kind resolves to a pill at both ends.
    anchors = drawn | {card.key for card in _cards_of(cards)}
    links = tuple(
        link
        for link in _links(spec, presentation_keys, context) + hoist_links
        if link.source_key in anchors and link.target_key in anchors
    )
    constraints = _constraints(spec, cards, context)

    return Presentation(
        cards=tuple(cards),
        links=links,
        constraints=constraints,
        legend=_legend(context, constraints),
        footer=_footer(spec, hidden_fixed),
        hidden_fixed=hidden_fixed,
    )
