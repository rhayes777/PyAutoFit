"""
Semantic extraction of a model's structure -- the ``graph_spec``.

Three layers, one of them here
------------------------------

The model-figure machinery is deliberately split into three layers, and this
module is the **first** of them:

1. **Semantic extraction** (*this module*) -- paths, object identity, parameter
   states, relationships and provenance, as plain frozen dataclasses. It knows
   nothing about pixels, fonts, colours, grouping-for-looks or detail levels,
   and it imports no drawing library (there is a test that asserts importing
   this module does not import ``matplotlib``).
2. **Presentation transformation** -- grouping (plates), omission, ordering and
   detail level.
3. **Layout and rendering** -- measurement, placement, routing and drawing.

Nothing below layer 1 may leak upward into it: a renderer consumes a
:class:`GraphSpec`, and a :class:`GraphSpec` never consults a renderer.

The ordering contract
---------------------

**Visual order is the order of** ``model.info``'s **parameter-detail section**,
which is the walk order of::

    model.path_instance_tuples_for_class(..., ignore_children=True)

i.e. plain ``__dict__`` (declaration) order, depth first.  Children of a
component and rows within a component are emitted in exactly that order, and a
renderer is forbidden from reordering them for packing.

``model.info`` prints *two* sections in two different orders, so the choice
matters.  The other candidate -- the structural summary ``model.parameterization``
-- is **rejected**: it reorders its lines through ``find_groups(..., limit=0)``,
which collapses sibling paths by suffix and therefore does not preserve
declaration order (this is exactly the defect that floated ``shear`` above
``mass`` in the prototype).

The ``model.info`` correspondence contract
------------------------------------------

    Every displayed model element resolves to its corresponding path or grouped
    paths in ``model.info``; every omission and every added annotation
    (``solved``, ``missing``) is explicit.

This is *not* a line-for-line claim.  :attr:`GraphSpec.path_index` is the
machine-readable half of it: a mapping from a stable element key to the tuple of
``model.info`` paths that element resolves to.  An element that is **not** in
``model.info`` -- a ``solved`` row, a ``latent`` row, an assertion -- is recorded
with an empty tuple and carries ``in_model_info=False``.  A ``missing`` row *is*
in ``model.info`` (it is the line ``Prior Missing: Enter Manually or Add to
Config``) and so keeps ``in_model_info=True``.

A **plate** stands for many paths at once, and the figure's partition can be
*finer* than ``model.info``'s grouping: the MGE splits into two 30-member
plates on ``ell_comps`` while ``model.info`` groups the shared ``centre`` of all
sixty as ``0 - 59``.  That mapping is recorded rather than hidden -- such an
entry is a ``{"figure": [...], "info": [...]}`` dict of ``"/"``-joined paths
instead of the plain tuple.  Both shapes are JSON-stable; see
:func:`_path_index`.

Properties, not one exclusive kind
----------------------------------

A :class:`ParamRow` carries **independent properties** rather than a single
``kind``.  A shared prior is still sampled; a tuple is still free.  So:

``sampling``
    ``free`` (a :class:`Prior` leaf), ``fixed`` (a ``Constant`` / plain
    ``float`` / plain ``int`` / plain ``tuple`` leaf, or a raw instance),
    ``solved`` (a quantity the fit determines that is absent from the model --
    supplied by ``solved_paths=`` or by an analysis' latent catalogue; phase 3
    supplies the domain rules) or ``missing`` (a required configuration value
    that is unset, i.e. a ``ConfigException`` sitting in the model tree, which
    ``model.info`` prints as *Prior Missing: Enter Manually or Add to Config*).
``sharing``
    ``prior_id`` plus :attr:`ParamRow.occurrences`, *every* path at which the
    same ``Prior`` object appears, and :attr:`ParamRow.direct_occurrences`, the
    subset of those that do not pass through a relation.
    :attr:`ParamRow.shared` is derived from the latter.  **Sharing is never a
    sampling state.**
``dimensionality``
    ``scalar`` or ``tuple``.  A tuple row carries :attr:`ParamRow.components`,
    one :class:`ParamRow` per slot (``centre_0``, ``centre_1``, ...), each with
    its own sampling status, ``prior_id`` and occurrences -- so a partially
    fixed or partially shared tuple is representable as a mixed state rather
    than being flattened.  A **fixed tuple constant** (a plain ``tuple`` of
    floats, e.g. ``centre=(0.1, 0.2)``) is *also* a tuple row, with fixed
    components; it is never dropped.
``provenance``
    :class:`Provenance`, see below.

Provenance kinds
----------------

``config-default``
    A prior that came from the prior configuration.  **Known gap:** phase 1
    cannot distinguish a user-set prior from the configured default -- nothing
    on a ``Prior`` records where it came from -- so *every* prior is emitted as
    ``config-default``.  ``user-prior`` is reserved for when a config diff (or a
    provenance field on ``Prior``) makes the distinction available.
``relation``
    A ``CompoundPrior`` / ``ModifiedPrior``: carries its defining
    ``expression`` (e.g. ``"normalization + sigma"``) and its ``operands``.
``latent``
    A row derived from ``type(analysis).Latent.keys(analysis)``.  Not part of
    the model, hence ``in_model_info=False``.
``user-prior``, ``assertion``, ``hierarchical-draw``, ``observed``
    **Reserved and not emitted by phase 1.**  ``assertion`` operands are carried
    by :class:`AssertionEdge` (assertions are edges, never rows);
    ``hierarchical-draw`` is phase 4 (a ``_HierarchicalFactor`` draw is *not* a
    sharing marker); ``observed`` is phase 4/5 (observed data must not be
    encoded as ``fixed``).

Relations and the ``sampling`` axis
-----------------------------------

The sampling axis is fixed to ``free | fixed | solved | missing`` and a relation
is modelled under *provenance*, so a relation row still needs a sampling status.
The rule: a relation is ``free`` when **any** operand is a free prior (its value
varies during sampling) and ``fixed`` when **every** operand is a constant.

Sharing and relation operands
-----------------------------

``all_paths_prior_tuples`` is the sharing detector, and it reports *every* path
at which a prior object sits.  A relation's operands are real attributes of the
``CompoundPrior``, so ``m.centre = m.normalization + m.sigma`` puts prior *n* at
both ``('normalization',)`` and ``('centre', 'self')``.  Both paths are genuine
and both are kept, in :attr:`ParamRow.occurrences`.

But a prior used once directly and once *inside a relation* is **related, not
shared**: the second path is the relation's own operand slot, not a second use
of the parameter, and counting it as sharing would draw a sharing edge for every
relation and would split plates that are genuine replicates.  So
:attr:`ParamRow.direct_occurrences` drops every occurrence whose path passes
through a relation object, and it -- not ``occurrences`` -- is what
:attr:`ParamRow.shared`, :class:`SharedEdge` emission,
``counts["shared_priors"]`` and rule R2's partition read.  The relation itself is
carried as a :class:`RelationEdge`.

Collapse: rules R1 and R2, and the safety condition
---------------------------------------------------

``GraphSpec.from_model(model, collapse=True)`` (the default) collapses repeated
sibling components into **plates**, recursively and bottom-up, over every node's
``children``.  Only ``kind="model"`` siblings collapse: a ``Collection`` is a
*frame*, not a repeated component -- but its model children do.  Every rule reads
the **uncollapsed** subtrees, because a plate hides the very prior ids and rows
the rules partition on.

**R1, the soft-plate signature.**  A member's signature is the recursive tuple
of its class name and kind, each row's ``(name, sampling, dimensionality, prior
class, provenance kind, prior configuration)`` -- and the same per component for
a tuple row -- and its children's signatures in order.  The prior's
configuration comes from its own public attributes (``__identifier_fields__``
plus its limits), never from ``repr``.  Prior **identity**, prior ``_label``
(which carries a per-instance counter) and constant **values** are all ignored,
so Gaussians differing only in a fixed ``sigma`` are still plate-mates.  Equal
signatures make candidate plate-mates.

**R2, the shared split.**  Within a candidate plate, each member is keyed by the
set of prior ids it shares with at least one *other* member.  Ids present in
**every** member do not discriminate (the MGE centre) and are dropped from every
key; a prior shared only *inside* one member is not cross-member at all and
never reaches the key -- without that qualifier the group model's eight extra
galaxies fall back to eight boxes.  The members are then partitioned by the
remaining key, which is what gives ``Gaussian x30`` + ``Gaussian x30`` rather
than one ``x60``.

**The safety condition** (Codex review point 7, mandatory).  A plate must
preserve **sharing, relations, assertions and exceptions** across its members.
A member also leaves the plate when its relation set (:class:`RelationEdge` s
whose target is inside it), its assertion set (:class:`AssertionEdge` s touching
it) or its external-sharing profile (the prior ids it shares with anything
*outside* the plate, keyed by the row path relative to the member) differs from
the others'.  Relations and assertions are keyed by their footprint *relative to
the member*, so eight galaxies carrying the same relation stay together while
one carrying it alone does not.  Exceptions -- the ``missing`` state -- are
already carried by R1, which compares every row's ``sampling``.  A member that
leaves is emitted as its own node, in declaration order.

A plate of two or more members becomes **one** :class:`ComponentNode` -- the
first member, rows and children -- carrying a :class:`PlateInfo`.  Its
``representative_key`` comes from ``find_groups`` over the member paths, so it
reads exactly as ``model.info`` prints it (``"0 - 29"``); the collapse never
invents a second notion of sameness.  Nothing depends on set iteration order:
every grouping is keyed by declaration order, so the output is byte-stable.

``counts`` reconcile both trees: ``components`` after collapse,
``components_raw`` before, ``plates`` how many of the former stand for more than
one of the latter.  The row-derived counts (``fixed_leaf_slots``, ``missing``)
are always of the **uncollapsed** tree -- collapsing must never make a fixed
value look absent from the model.

Traps this module already handles
---------------------------------

* ``TuplePrior`` is **not** an ``AbstractPriorModel``: a walk over
  ``direct_prior_tuples`` + ``direct_prior_model_tuples`` silently loses every
  tuple parameter.  Tuple priors are extracted explicitly.
* ``CompoundPrior`` / ``ModifiedPrior`` / ``ComparisonAssertion`` **are**
  ``AbstractPriorModel`` s with ``cls = float``, so they masquerade as
  components.  They are filtered out of the component walk.
* ``repr()`` of a ``CompoundPrior`` subclass without ``__str__`` recurses
  forever (``compound.py``).  **Nothing here ever calls ``str()`` or ``repr()``
  on a compound prior or an assertion** -- expressions are built from a symbol
  table.
* ``Model(int)`` / ``Model(float)`` wrappers (e.g. ``Delaunay.pixels``) are rows
  on their owner, never components (rule R7).
* ``gathered_assertions()`` is a **method**; ``assertions`` is a property that
  only sees the model's own list.
* ``obj_id`` is the ``ModelObject`` id counter, not ``id(obj)``: a memory
  address is not stable across processes and would break the determinism
  contract.

Entry points
------------

``GraphSpec.from_model(model, analysis=None, collapse=True, solved_paths=())``
and the module-level ``graph_spec_from(model, **kwargs)``.  ``collapse=False``
returns the uncollapsed tree -- the tree the construct catalogue is asserted
against; ``collapse=True`` (the default) applies :func:`_collapse_siblings`.
"""

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from autonerves.exc import ConfigException

from autofit.mapper.prior.abstract import Prior
from autofit.mapper.prior.arithmetic.assertion import (
    ComparisonAssertion,
    CompoundAssertion,
    GreaterThanLessThanAssertion,
    GreaterThanLessThanEqualAssertion,
)
from autofit.mapper.prior.arithmetic.compound import (
    AbsolutePrior,
    CompoundPrior,
    DivisionPrior,
    FloorDivPrior,
    Log,
    Log10,
    ModPrior,
    ModifiedPrior,
    MultiplePrior,
    NegativePrior,
    PowerPrior,
    SumPrior,
)
from autofit.mapper.prior.constant import Constant
from autofit.mapper.prior.tuple_prior import TuplePrior
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.mapper.prior_model.collection import Collection
from autofit.mapper.prior_model.prior_model import Model
from autofit.mapper.prior_model.representative import (
    find_groups,
    integers_representative_key,
)

__all__ = [
    "Path",
    "Provenance",
    "ParamRow",
    "PlateInfo",
    "ComponentNode",
    "SharedEdge",
    "RelationEdge",
    "AssertionEdge",
    "GraphSpec",
    "graph_spec_from",
]

Path = Tuple[str, ...]

#: Attribute names that are never parameter slots.
_SKIP_ATTRIBUTES = frozenset({"id", "item_number", "cls"})

#: Per-class extra skips, matched anywhere in a component's MRO by class name
#: (so no import of the ``graphical`` layer is needed here). ``Array`` carries
#: ``shape`` / ``indices`` as bookkeeping, and ``GlobalPriorModel`` carries the
#: declarative factor graph it was built from; none of the three is a slot.
_SKIP_BY_CLASS_NAME = {
    "Array": frozenset({"shape", "indices"}),
    "GlobalPriorModel": frozenset({"factor"}),
}

#: Binary compound priors, mapped to their infix symbol.
_BINARY_SYMBOLS = {
    SumPrior: "+",
    MultiplePrior: "*",
    DivisionPrior: "/",
    FloorDivPrior: "//",
    ModPrior: "%",
    PowerPrior: "**",
}

#: Unary modified priors, mapped to a format template.
_UNARY_TEMPLATES = {
    NegativePrior: "-({})",
    AbsolutePrior: "abs({})",
    Log: "log({})",
    Log10: "log10({})",
}

#: Assertion classes, mapped to the operator they assert.
_ASSERTION_OPERATORS = {
    GreaterThanLessThanAssertion: "<",
    GreaterThanLessThanEqualAssertion: "<=",
    CompoundAssertion: "and",
}

_RELATION_CLASSES = (CompoundPrior, ModifiedPrior)
_NOT_A_COMPONENT = (
    CompoundPrior,
    ModifiedPrior,
    ComparisonAssertion,
    CompoundAssertion,
)


# ----------------------------------------------------------------------------
# data model
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Provenance:
    """
    Where a parameter slot's value came from.

    Parameters
    ----------
    kind
        One of ``config-default``, ``user-prior``, ``relation``, ``assertion``,
        ``hierarchical-draw``, ``observed``, ``latent``.  Phase 1 emits only
        ``config-default``, ``relation`` and ``latent``; the rest are reserved
        (see the module docstring).
    expression
        For ``relation``, the defining expression, e.g. ``"sigma * 2.0"``.
    operands
        For ``relation``, the operand names -- a dotted ``model.info`` path when
        the operand is a prior found in the model, otherwise the attribute name
        the compound prior recorded for it, otherwise the rendered constant.
    """

    kind: str
    expression: Optional[str] = None
    operands: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "expression": self.expression,
            "operands": list(self.operands),
        }


@dataclass(frozen=True)
class ParamRow:
    """
    One parameter slot.

    See the module docstring for the property axes.  ``occurrences`` is every
    path at which this row's ``Prior`` object appears; ``shared`` is derived from
    it and is *never* a sampling state.
    """

    name: str
    path: Path
    sampling: str
    prior_id: Optional[int] = None
    occurrences: Tuple[Path, ...] = ()
    direct_occurrences: Tuple[Path, ...] = ()
    dimensionality: str = "scalar"
    components: Tuple["ParamRow", ...] = ()
    provenance: Provenance = field(default_factory=lambda: Provenance("config-default"))
    value: Any = None
    prior_cls_name: Optional[str] = None
    in_model_info: bool = True
    is_instance: bool = False

    @property
    def shared(self) -> bool:
        """
        Whether this row's prior object appears at more than one **direct** path.

        Derived from :attr:`direct_occurrences`, not :attr:`occurrences`: a prior
        used once directly and once as the operand of a relation is *related*,
        not shared (see the module docstring, "Sharing and relation operands").
        """
        return len(self.direct_occurrences) > 1

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": list(self.path),
            "sampling": self.sampling,
            "prior_id": self.prior_id,
            "occurrences": [list(path) for path in self.occurrences],
            "direct_occurrences": [list(path) for path in self.direct_occurrences],
            "shared": self.shared,
            "dimensionality": self.dimensionality,
            "components": [component.to_dict() for component in self.components],
            "provenance": self.provenance.to_dict(),
            "value": self.value,
            "prior_cls_name": self.prior_cls_name,
            "in_model_info": self.in_model_info,
            "is_instance": self.is_instance,
        }


@dataclass(frozen=True)
class PlateInfo:
    """
    What a collapsed plate repeats.

    Parameters
    ----------
    count
        How many components the plate stands for.
    member_paths
        The path of every member, in declaration order.  The first is the
        representative whose rows and children the plate's
        :class:`ComponentNode` carries.
    representative_key
        The key ``find_groups`` puts in place of the member index, so it reads
        exactly as ``model.info`` prints it (``"0 - 29"``).
    repeats
        One line naming what is repeated, built mechanically from the
        representative's own rows, e.g. ``"Gaussian with priors centre
        \u21c4 shared, sigma fixed (varies by member)"``.
    shared_in_all
        Prior **ids** carried by *every* member (these do not discriminate, so
        they never split a plate -- the MGE centre).  Ascending.
    varies_by_member
        Dotted row paths, relative to a member, whose fixed value differs
        between members.
    """

    count: int
    member_paths: Tuple[Path, ...] = ()
    representative_key: Optional[str] = None
    repeats: Tuple[str, ...] = ()
    shared_in_all: Tuple[int, ...] = ()
    varies_by_member: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "member_paths": [list(path) for path in self.member_paths],
            "representative_key": self.representative_key,
            "repeats": list(self.repeats),
            "shared_in_all": list(self.shared_in_all),
            "varies_by_member": list(self.varies_by_member),
        }


@dataclass(frozen=True)
class ComponentNode:
    """
    One ``Model`` / ``Collection`` / ``GlobalPriorModel`` in the nesting tree.

    ``name`` is the exact attribute spelling, or the list index as a string
    (``"0"``, ``"1"``) for a list-built ``Collection``.  The root node has
    ``path=()`` and ``name="model"``.  ``obj_id`` is the ``ModelObject`` id
    counter: the same ``obj_id`` at several paths is how a component shared
    across datasets is detected.
    """

    path: Path
    name: str
    cls_name: str
    kind: str
    obj_id: Optional[int]
    rows: Tuple[ParamRow, ...] = ()
    children: Tuple["ComponentNode", ...] = ()
    plate: Optional[PlateInfo] = None

    def to_dict(self) -> dict:
        return {
            "path": list(self.path),
            "name": self.name,
            "cls_name": self.cls_name,
            "kind": self.kind,
            "obj_id": self.obj_id,
            "rows": [row.to_dict() for row in self.rows],
            "children": [child.to_dict() for child in self.children],
            "plate": self.plate.to_dict() if self.plate is not None else None,
        }


@dataclass(frozen=True)
class SharedEdge:
    """One ``Prior`` object appearing at more than one path."""

    prior_id: int
    occurrences: Tuple[Path, ...]

    def to_dict(self) -> dict:
        return {
            "prior_id": self.prior_id,
            "occurrences": [list(path) for path in self.occurrences],
        }


@dataclass(frozen=True)
class RelationEdge:
    """A ``CompoundPrior`` / ``ModifiedPrior`` and the operands it is built from."""

    target_path: Path
    expression: str
    operand_paths: Tuple[Path, ...] = ()
    prior_id: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "target_path": list(self.target_path),
            "expression": self.expression,
            "operand_paths": [list(path) for path in self.operand_paths],
            "prior_id": self.prior_id,
        }


@dataclass(frozen=True)
class AssertionEdge:
    """
    An assertion attached anywhere in the model tree.

    Assertions are **not** part of the tree and are **not** in ``model.info``;
    they are edges only.
    """

    op: str
    left: str
    right: str
    name: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "op": self.op,
            "left": self.left,
            "right": self.right,
            "name": self.name,
        }


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def _as_path(path: Iterable) -> Path:
    """Normalise a model path to a tuple of strings (list indices become ``"0"``)."""
    return tuple(str(step) for step in path)


def _dotted(path: Path) -> str:
    return ".".join(path)


def _key(path: Path) -> str:
    """The stable element key used by :attr:`GraphSpec.path_index`."""
    return "/".join(path)


def _numeric(value) -> Optional[float]:
    """The float behind a ``Constant`` / ``float`` / ``int``, else ``None``."""
    if isinstance(value, Constant):
        return float(value.value)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _is_dropped_model(obj) -> bool:
    """
    Rule R7 -- ``Model(int)`` / ``Model(float)`` is a row on its owner, never a
    component of its own.
    """
    return isinstance(obj, Model) and getattr(obj, "cls", None) in (int, float)


def _cls_name(obj) -> str:
    """The displayed class name of a component."""
    cls = getattr(obj, "cls", None)
    if cls is not None:
        # `Model(function)` keeps the function itself in `cls`, and a function
        # has a `__name__` just as a class does.
        name = getattr(cls, "__name__", None)
        if name is not None:
            return name
    return type(obj).__name__


def _kind(obj) -> str:
    """``model`` / ``collection`` / ``global``."""
    if type(obj).__name__ == "GlobalPriorModel":
        return "global"
    if isinstance(obj, Collection):
        return "collection"
    return "model"


def _skipped_attributes(obj) -> frozenset:
    """Attribute names that are not parameter slots on this component."""
    skipped = set(_SKIP_ATTRIBUTES)
    for cls in type(obj).__mro__:
        skipped |= _SKIP_BY_CLASS_NAME.get(cls.__name__, frozenset())
    # Honour `__exclude_identifier_fields__` exactly as `AbstractPriorModel.info`
    # does: attributes a class has already declared "not part of the model's
    # identity" do not leak into the figure either.
    skipped |= set(getattr(type(obj), "__exclude_identifier_fields__", ()) or ())
    names = getattr(type(obj), "_cached_property_names", None)
    if callable(names):
        try:
            skipped |= set(names())
        except Exception:  # pragma: no cover - defensive
            pass
    return frozenset(skipped)


def _attribute_items(obj) -> List[Tuple[str, Any]]:
    """
    The component's direct attributes, in declaration (``__dict__``) order.

    This is the union of ``direct_prior_tuples``, ``direct_tuple_priors`` /
    ``tuple_prior_tuples`` and ``direct_instance_tuples`` -- read straight off
    ``__dict__`` so that (a) declaration order is preserved, (b) the known
    double-count of ``Constant`` s in ``direct_instance_tuples`` (a ``Constant``
    is a ``float``, so it is returned by both halves of that property) cannot
    happen, since each attribute name is visited exactly once, and (c) raw
    instance leaves -- which no ``direct_*`` property reports -- are seen.
    """
    skipped = _skipped_attributes(obj)
    return [
        (name, value)
        for name, value in obj.__dict__.items()
        if not name.startswith("_") and name not in skipped
    ]


# ----------------------------------------------------------------------------
# relation expressions
# ----------------------------------------------------------------------------


def _operand_name(value, model, fallback: Optional[str]) -> str:
    """
    Render one operand of a compound prior.

    A prior that is in the model renders as its dotted path; any other prior
    renders as the attribute name the compound recorded for it; a constant
    renders as its float.  ``str()`` / ``repr()`` is never called on a compound
    prior or an assertion.
    """
    if isinstance(value, Prior):
        path = model.path_for_prior(value)
        if path is not None:
            return _dotted(_as_path(path))
        return fallback or f"prior_{value.id}"
    number = _numeric(value)
    if number is not None:
        return repr(number)
    if fallback:
        return fallback
    return type(value).__name__


def _expression(obj, model, top: bool = True) -> str:
    """
    The defining expression of a compound / modified prior, built from a symbol
    table.  Never calls ``str()``/``repr()`` on a compound (which recurses
    forever, ``compound.py``).
    """
    for cls, symbol in _BINARY_SYMBOLS.items():
        if isinstance(obj, cls):
            left = _render_operand(obj._left, model, getattr(obj, "_left_name", None))
            right = _render_operand(
                obj._right, model, getattr(obj, "_right_name", None)
            )
            rendered = f"{left} {symbol} {right}"
            return rendered if top else f"({rendered})"
    for cls, template in _UNARY_TEMPLATES.items():
        if isinstance(obj, cls):
            inner = _render_operand(
                getattr(obj, obj._prior_name, None),
                model,
                getattr(obj, "_prior_name", None),
            )
            return template.format(inner)
    if isinstance(obj, ComparisonAssertion):
        operator = _ASSERTION_OPERATORS.get(type(obj), type(obj).__name__)
        left = _render_operand(obj._left, model, getattr(obj, "_left_name", None))
        right = _render_operand(obj._right, model, getattr(obj, "_right_name", None))
        return f"{left} {operator} {right}"
    if isinstance(obj, CompoundAssertion):
        left = _expression(obj.assertion_1, model, top=False)
        right = _expression(obj.assertion_2, model, top=False)
        return f"{left} and {right}"
    # An unrecognised compound subclass: name it rather than risk `repr`.
    return type(obj).__name__


def _render_operand(value, model, fallback: Optional[str]) -> str:
    if isinstance(value, _RELATION_CLASSES + (ComparisonAssertion, CompoundAssertion)):
        return _expression(value, model, top=False)
    return _operand_name(value, model, fallback)


def _operand_pairs(obj, model) -> List[Tuple[str, Any]]:
    """
    ``(rendered name, operand)`` for every leaf operand of a compound, left to
    right, recursing through nested compounds.
    """
    pairs: List[Tuple[str, Any]] = []

    def _walk(node, fallback):
        if isinstance(node, _RELATION_CLASSES):
            if isinstance(node, ModifiedPrior):
                _walk(
                    getattr(node, node._prior_name, None),
                    getattr(node, "_prior_name", None),
                )
                return
            _walk(node._left, getattr(node, "_left_name", None))
            _walk(node._right, getattr(node, "_right_name", None))
            return
        pairs.append((_operand_name(node, model, fallback), node))

    _walk(obj, None)
    return pairs


def _relation_provenance(obj, model) -> Tuple[Provenance, Tuple[Path, ...], str]:
    """The provenance, prior operand paths and sampling status of a relation."""
    expression = _expression(obj, model)
    pairs = _operand_pairs(obj, model)

    operands: List[str] = []
    operand_paths: List[Path] = []
    has_free = False
    for name, operand in pairs:
        if name not in operands:
            operands.append(name)
        if isinstance(operand, Prior):
            has_free = True
            path = model.path_for_prior(operand)
            if path is not None:
                path = _as_path(path)
                if path not in operand_paths:
                    operand_paths.append(path)

    sampling = "free" if has_free else "fixed"
    return (
        Provenance("relation", expression, tuple(operands)),
        tuple(operand_paths),
        sampling,
    )


# ----------------------------------------------------------------------------
# extraction
# ----------------------------------------------------------------------------


class _Extractor:
    def __init__(self, model, analysis=None, solved_paths: Sequence = ()):
        self.model = model
        self.analysis = analysis
        self.solved_paths = {
            _as_path(path.split(".")) if isinstance(path, str) else _as_path(path)
            for path in (solved_paths or ())
        }

        #: Every path a prior sits at, and the subset of those that do not pass
        #: through a relation object (see the module docstring).
        self.occurrences: Dict[int, Tuple[Path, ...]] = {}
        self.direct_occurrences: Dict[int, Tuple[Path, ...]] = {}
        #: ``prior id -> the prior's configuration arguments``, for rule R1.
        self.prior_config: Dict[int, Tuple] = {}

        self.relation_paths = frozenset(
            _as_path(path)
            for path, _ in model.path_instance_tuples_for_class(_RELATION_CLASSES)
        )
        for paths, prior in model.all_paths_prior_tuples:
            paths = tuple(_as_path(path) for path in paths)
            self.occurrences[prior.id] = paths
            self.direct_occurrences[prior.id] = tuple(
                path
                for path in paths
                if not _passes_through_relation(path, self.relation_paths)
            )
            self.prior_config[prior.id] = _prior_configuration(prior)

        self._known_component_paths = set()

    # -- components ---------------------------------------------------------

    def _component_paths(self) -> List[Path]:
        """
        The component walk, in its returned order, with the root included and
        the non-components filtered out.
        """
        raw = self.model.path_instance_tuples_for_class(
            (Model, Collection), ignore_children=False
        )
        entries: List[Tuple[Path, Any]] = [(_as_path(path), obj) for path, obj in raw]
        if not entries or entries[0][0] != ():
            entries.insert(0, ((), self.model))

        kept: List[Tuple[Path, Any]] = []
        dropped: List[Path] = []
        for path, obj in entries:
            if any(path[: len(prefix)] == prefix for prefix in dropped):
                dropped.append(path)
                continue
            if path != () and isinstance(obj, _NOT_A_COMPONENT):
                dropped.append(path)
                continue
            if path != () and _is_dropped_model(obj):
                dropped.append(path)
                continue
            kept.append((path, obj))
        self._known_component_paths = {path for path, _ in kept}
        return kept

    def build_root(self) -> ComponentNode:
        entries = self._component_paths()
        by_path = {path: obj for path, obj in entries}
        children_of: Dict[Path, List[Path]] = {path: [] for path, _ in entries}
        for path, _ in entries:
            if path == ():
                continue
            for length in range(len(path) - 1, -1, -1):
                parent = path[:length]
                if parent in children_of:
                    children_of[parent].append(path)
                    break
        return self._node(
            (), by_path[()], "model", by_path=by_path, children_of=children_of
        )

    def _node(self, path: Path, obj, name: str, by_path, children_of) -> ComponentNode:
        rows: List[ParamRow] = []
        children: List[ComponentNode] = []
        child_paths = list(children_of.get(path, ()))

        for attribute, value in _attribute_items(obj):
            child_path = path + (attribute,)
            if child_path in by_path:
                children.append(
                    self._node(
                        child_path,
                        by_path[child_path],
                        attribute,
                        by_path=by_path,
                        children_of=children_of,
                    )
                )
                if child_path in child_paths:
                    child_paths.remove(child_path)
                continue
            row = self._row(attribute, child_path, value)
            if row is not None:
                rows.append(row)
                continue
            if isinstance(value, AbstractPriorModel):
                # An `AbstractPriorModel` that the mandated `(Model, Collection)`
                # walk cannot see -- an `af.Array`, say -- is promoted to a
                # component here rather than dropped, so no parameter is lost.
                promoted = {
                    p: o for p, o in self._promoted_entries(child_path, value).items()
                }
                by_path.update(promoted)
                sub_children: Dict[Path, List[Path]] = {p: [] for p in promoted}
                for p in promoted:
                    if p == child_path:
                        continue
                    for length in range(len(p) - 1, -1, -1):
                        parent = p[:length]
                        if parent in sub_children:
                            sub_children[parent].append(p)
                            break
                merged = dict(children_of)
                merged.update(sub_children)
                children.append(
                    self._node(
                        child_path,
                        value,
                        attribute,
                        by_path=by_path,
                        children_of=merged,
                    )
                )

        # Any component the walk found under this node whose owning attribute
        # was not visible in `__dict__` (a list entry, say) still belongs here.
        for child_path in child_paths:
            children.append(
                self._node(
                    child_path,
                    by_path[child_path],
                    child_path[-1],
                    by_path=by_path,
                    children_of=children_of,
                )
            )

        return ComponentNode(
            path=path,
            name=name,
            cls_name=_cls_name(obj),
            kind=_kind(obj),
            obj_id=getattr(obj, "id", None),
            rows=tuple(rows),
            children=tuple(children),
        )

    def _promoted_entries(self, path: Path, obj) -> Dict[Path, Any]:
        entries = {path: obj}
        for sub_path, sub_obj in obj.path_instance_tuples_for_class(
            (Model, Collection), ignore_children=False
        ):
            sub_path = _as_path(sub_path)
            if sub_path == ():
                continue
            if isinstance(sub_obj, _NOT_A_COMPONENT) or _is_dropped_model(sub_obj):
                continue
            entries[path + sub_path] = sub_obj
        return entries

    # -- rows ---------------------------------------------------------------

    def _occurrences(self, prior) -> Tuple[Path, ...]:
        return self.occurrences.get(prior.id, ())

    def _direct_occurrences(self, prior) -> Tuple[Path, ...]:
        return self.direct_occurrences.get(prior.id, ())

    def _solved(self, row: ParamRow) -> ParamRow:
        if row.path in self.solved_paths:
            return replace(row, sampling="solved", in_model_info=False)
        return row

    def _row(self, name: str, path: Path, value) -> Optional[ParamRow]:
        """
        A row for a direct attribute, or ``None`` when the attribute is not a
        parameter slot (i.e. it is a child component).
        """
        if isinstance(value, Prior):
            return self._solved(
                ParamRow(
                    name=name,
                    path=path,
                    sampling="free",
                    prior_id=value.id,
                    occurrences=self._occurrences(value),
                    direct_occurrences=self._direct_occurrences(value),
                    prior_cls_name=type(value).__name__,
                )
            )

        if isinstance(value, TuplePrior):
            components = tuple(
                self._tuple_component(path + (sub_name,), sub_name, sub_value)
                for sub_name, sub_value in sorted(
                    (
                        (k, v)
                        for k, v in value.__dict__.items()
                        if not k.startswith("_") and k != "id"
                    ),
                    key=lambda item: item[0],
                )
            )
            return self._solved(
                ParamRow(
                    name=name,
                    path=path,
                    sampling=_tuple_sampling(components),
                    dimensionality="tuple",
                    components=components,
                    prior_cls_name="TuplePrior",
                )
            )

        if isinstance(value, _RELATION_CLASSES):
            provenance, _, sampling = _relation_provenance(value, self.model)
            return self._solved(
                ParamRow(
                    name=name,
                    path=path,
                    sampling=sampling,
                    provenance=provenance,
                    prior_cls_name=type(value).__name__,
                )
            )

        if isinstance(value, ConfigException):
            # Exactly what `model.info` prints as
            # "Prior Missing: Enter Manually or Add to Config".
            return ParamRow(
                name=name,
                path=path,
                sampling="missing",
                prior_cls_name="ConfigException",
            )

        if isinstance(value, tuple):
            components = tuple(
                ParamRow(
                    name=f"{name}_{index}",
                    path=path + (f"{name}_{index}",),
                    sampling="fixed",
                    value=_numeric(item),
                    prior_cls_name=type(item).__name__,
                )
                for index, item in enumerate(value)
            )
            return self._solved(
                ParamRow(
                    name=name,
                    path=path,
                    sampling="fixed",
                    dimensionality="tuple",
                    components=components,
                    value=[_numeric(item) for item in value],
                    prior_cls_name="tuple",
                )
            )

        number = _numeric(value)
        if number is not None:
            return self._solved(
                ParamRow(
                    name=name,
                    path=path,
                    sampling="fixed",
                    value=number,
                    prior_cls_name=type(value).__name__,
                )
            )

        if _is_dropped_model(value):
            return self._solved(self._int_model_row(name, path, value))

        if isinstance(value, AbstractPriorModel):
            return None

        if value is None:
            # An unset optional attribute (e.g. ``Basis(regularization=None)``).
            # ``model.info`` prints nothing for it, so it is not a slot at all --
            # emitting a row would violate the correspondence contract.
            return None

        # A raw Python object assigned into a Collection/Model.
        return self._solved(
            ParamRow(
                name=name,
                path=path,
                sampling="fixed",
                prior_cls_name=type(value).__name__,
                is_instance=True,
            )
        )

    def _tuple_component(self, path: Path, name: str, value) -> ParamRow:
        if isinstance(value, Prior):
            return ParamRow(
                name=name,
                path=path,
                sampling="free",
                prior_id=value.id,
                occurrences=self._occurrences(value),
                direct_occurrences=self._direct_occurrences(value),
                prior_cls_name=type(value).__name__,
            )
        if isinstance(value, _RELATION_CLASSES):
            provenance, _, sampling = _relation_provenance(value, self.model)
            return ParamRow(
                name=name,
                path=path,
                sampling=sampling,
                provenance=provenance,
                prior_cls_name=type(value).__name__,
            )
        if isinstance(value, ConfigException):
            return ParamRow(
                name=name,
                path=path,
                sampling="missing",
                prior_cls_name="ConfigException",
            )
        return ParamRow(
            name=name,
            path=path,
            sampling="fixed",
            value=_numeric(value),
            prior_cls_name=type(value).__name__,
        )

    def _int_model_row(self, name: str, path: Path, value) -> ParamRow:
        """
        Rule R7 -- an ``af.Model(int)`` / ``af.Model(float)`` wrapper is a row on
        its owner.  Its single prior, constant or ``ConfigException``, if it has
        one, *is* that row; a wrapper with no leaf at all carries no value and is
        absent from ``model.info``, so it is recorded with ``value=None`` and
        ``in_model_info=False`` rather than dropped.
        """
        cls_name = _cls_name(value)
        for attribute, leaf in _attribute_items(value):
            if isinstance(leaf, Prior):
                return ParamRow(
                    name=name,
                    path=path,
                    sampling="free",
                    prior_id=leaf.id,
                    occurrences=self._occurrences(leaf),
                    direct_occurrences=self._direct_occurrences(leaf),
                    prior_cls_name=type(leaf).__name__,
                )
            if isinstance(leaf, ConfigException):
                return ParamRow(
                    name=name,
                    path=path,
                    sampling="missing",
                    prior_cls_name="ConfigException",
                )
            number = _numeric(leaf)
            if number is not None:
                return ParamRow(
                    name=name,
                    path=path,
                    sampling="fixed",
                    value=number,
                    prior_cls_name=cls_name,
                )
        return ParamRow(
            name=name,
            path=path,
            sampling="fixed",
            value=None,
            prior_cls_name=cls_name,
            in_model_info=False,
        )

    # -- edges --------------------------------------------------------------

    def shared_edges(self) -> Tuple[SharedEdge, ...]:
        """
        One edge per prior that sits at more than one **direct** path.  A path
        that passes through a relation object is an operand of that relation,
        not a second use of the prior, and is carried by a
        :class:`RelationEdge` instead (module docstring, "Sharing and relation
        operands").
        """
        edges = []
        for _, prior in self.model.all_paths_prior_tuples:
            paths = self.direct_occurrences.get(prior.id, ())
            if len(paths) > 1:
                edges.append(SharedEdge(prior.id, paths))
        return tuple(edges)

    def relation_edges(self) -> Tuple[RelationEdge, ...]:
        edges = []
        for path, obj in self.model.path_instance_tuples_for_class(_RELATION_CLASSES):
            provenance, operand_paths, _ = _relation_provenance(obj, self.model)
            edges.append(
                RelationEdge(
                    target_path=_as_path(path),
                    expression=provenance.expression or "",
                    operand_paths=operand_paths,
                    prior_id=None,
                )
            )
        return tuple(edges)

    def assertion_edges(self) -> Tuple[AssertionEdge, ...]:
        edges = []
        for assertion in self.model.gathered_assertions():
            if assertion is True or assertion is False:
                continue
            operator = _ASSERTION_OPERATORS.get(type(assertion))
            if operator is None:
                for cls, symbol in _ASSERTION_OPERATORS.items():
                    if isinstance(assertion, cls):
                        operator = symbol
                        break
            if operator is None:
                operator = type(assertion).__name__
            if isinstance(assertion, CompoundAssertion):
                left = _expression(assertion.assertion_1, self.model, top=False)
                right = _expression(assertion.assertion_2, self.model, top=False)
            else:
                left = _render_operand(
                    getattr(assertion, "_left", None),
                    self.model,
                    getattr(assertion, "_left_name", None),
                )
                right = _render_operand(
                    getattr(assertion, "_right", None),
                    self.model,
                    getattr(assertion, "_right_name", None),
                )
            edges.append(
                AssertionEdge(
                    op=operator,
                    left=left,
                    right=right,
                    name=getattr(assertion, "_name", "") or None,
                )
            )
        return tuple(edges)

    # -- latents ------------------------------------------------------------

    def latent_keys(self) -> List[str]:
        if self.analysis is None:
            return []
        latent = getattr(type(self.analysis), "Latent", None)
        if latent is None:
            return []
        try:
            return list(latent.keys(self.analysis))
        except Exception:  # pragma: no cover - a broken analysis is not our error
            return []

    def attach_latents(self, root: ComponentNode) -> ComponentNode:
        keys = self.latent_keys()
        if not keys:
            return root
        by_path: Dict[Path, List[ParamRow]] = {}
        for key in keys:
            parts = tuple(key.split("."))
            owner, name = parts[:-1], parts[-1]
            by_path.setdefault(owner, []).append(
                ParamRow(
                    name=name,
                    path=owner + (name,),
                    sampling="solved",
                    provenance=Provenance("latent"),
                    in_model_info=False,
                )
            )
        known = _paths_of(root)

        def _attach(node: ComponentNode) -> ComponentNode:
            extra = list(by_path.get(node.path, ()))
            if node.path == ():
                for owner, rows in by_path.items():
                    if owner not in known:
                        extra.extend(rows)
            return replace(
                node,
                rows=node.rows + tuple(extra),
                children=tuple(_attach(child) for child in node.children),
            )

        return _attach(root)


def _tuple_sampling(components: Tuple[ParamRow, ...]) -> str:
    """
    A tuple row's own sampling status: ``free`` if any slot is sampled, else
    ``missing`` if any slot is unset, else ``fixed``.  The mixed state lives in
    the per-slot ``components``.
    """
    if any(component.sampling == "free" for component in components):
        return "free"
    if any(component.sampling == "missing" for component in components):
        return "missing"
    if any(component.sampling == "solved" for component in components):
        return "solved"
    return "fixed"


def _paths_of(node: ComponentNode) -> set:
    paths = {node.path}
    for child in node.children:
        paths |= _paths_of(child)
    return paths


# ----------------------------------------------------------------------------
# collapse -- rules R1 / R2 and the safety condition
# ----------------------------------------------------------------------------

#: Prior attributes that are configuration on every prior family, whatever
#: ``__identifier_fields__`` says.  ``_label`` (a per-instance counter) and the
#: prior id are deliberately absent: rule R1 ignores both.
_ALWAYS_CONFIGURATION = ("lower_limit", "upper_limit")


def _passes_through_relation(path: Path, relation_paths) -> bool:
    """Whether any proper prefix of ``path`` is a relation object."""
    return any(path[:length] in relation_paths for length in range(1, len(path)))


def _prior_configuration(prior) -> Tuple[Tuple[str, Any], ...]:
    """
    A prior's **configuration arguments** -- what rule R1 compares.

    Read from the prior's own public attributes (``__identifier_fields__``,
    which is exactly the set autofit already treats as a prior's identity, plus
    its limits).  Never ``repr``: a compound prior's ``repr`` recurses forever.
    """
    names: List[str] = []
    for name in getattr(type(prior), "__identifier_fields__", ()) or ():
        if name not in names:
            names.append(name)
    for name in _ALWAYS_CONFIGURATION:
        if name not in names:
            names.append(name)

    configuration: List[Tuple[str, Any]] = []
    for name in names:
        value = getattr(prior, name, None)
        number = _numeric(value)
        configuration.append((name, number if number is not None else None))
    return tuple(configuration)


def _assertion_prior_ids(assertion) -> frozenset:
    """
    The ids of every prior an assertion touches, walked from the symbol table
    rather than through ``repr`` (which recurses forever on a compound).
    """
    ids = set()

    def _walk(node):
        if isinstance(node, Prior):
            ids.add(node.id)
            return
        if isinstance(node, CompoundAssertion):
            _walk(node.assertion_1)
            _walk(node.assertion_2)
            return
        if isinstance(node, ModifiedPrior):
            _walk(getattr(node, node._prior_name, None))
            return
        if isinstance(node, _RELATION_CLASSES + (ComparisonAssertion,)):
            _walk(getattr(node, "_left", None))
            _walk(getattr(node, "_right", None))
            return

    _walk(assertion)
    return frozenset(ids)


class _CollapseContext:
    """Everything the collapse rules need that is not on the tree itself."""

    def __init__(self, extractor: "_Extractor"):
        self.direct_occurrences = extractor.direct_occurrences
        self.prior_config = extractor.prior_config
        self.relations = extractor.relation_edges()
        self.assertions = tuple(
            _assertion_prior_ids(assertion)
            for assertion in extractor.model.gathered_assertions()
            if assertion is not True and assertion is not False
        )


def _all_rows(node: ComponentNode, base: int = None):
    """
    ``(relative dotted path, row)`` for every row in a subtree, tuple slots
    included, in visual order.  Paths are relative to ``node``.
    """
    prefix = len(node.path) if base is None else base

    def _walk(current: ComponentNode):
        for row in current.rows:
            yield _dotted(row.path[prefix:]), row
            for component in row.components:
                yield _dotted(component.path[prefix:]), component
        for child in current.children:
            yield from _walk(child)

    return list(_walk(node))


def _prior_ids_in(node: ComponentNode) -> set:
    return {row.prior_id for _, row in _all_rows(node) if row.prior_id is not None}


def _row_signature(row: ParamRow, context: _CollapseContext) -> Tuple:
    """
    Rule R1's per-row signature: everything but prior **identity**, prior
    ``_label`` and constant **values**.
    """
    return (
        row.name,
        row.sampling,
        row.dimensionality,
        row.prior_cls_name,
        row.provenance.kind,
        row.in_model_info,
        row.is_instance,
        () if row.prior_id is None else context.prior_config.get(row.prior_id, ()),
        tuple(_row_signature(component, context) for component in row.components),
    )


def _signature(node: ComponentNode, context: _CollapseContext) -> Tuple:
    """
    Rule R1's **soft-plate signature**: the class tree and prior configuration,
    recursively, ignoring prior identity, prior ``_label`` and constant values.
    """
    return (
        node.cls_name,
        node.kind,
        tuple(_row_signature(row, context) for row in node.rows),
        tuple(_signature(child, context) for child in node.children),
    )


def _relative(path: Path, member: ComponentNode) -> Optional[Path]:
    """``path`` relative to ``member``, or ``None`` when it is outside it."""
    prefix = member.path
    if path[: len(prefix)] == prefix:
        return path[len(prefix) :]
    return None


def _safety_profile(
    member: ComponentNode,
    members: List[ComponentNode],
    context: _CollapseContext,
) -> Tuple:
    """
    The safety condition (Codex review point 7): a plate must preserve
    **sharing, relations, assertions and exceptions** across its members.

    Relations and assertions are keyed by their footprint *relative to the
    member*, so eight galaxies carrying the same relation stay together while a
    single galaxy that carries one on its own leaves.  External sharing is keyed
    by the relative row path, so a member that shares a prior with something
    outside the plate leaves it.  (Exceptions -- the ``missing`` state -- are
    already part of the R1 signature, which carries every row's ``sampling``.)
    """
    relations = []
    for edge in context.relations:
        target = _relative(edge.target_path, member)
        if target is None:
            continue
        operands = []
        for operand in edge.operand_paths:
            relative = _relative(operand, member)
            operands.append(
                relative if relative is not None else ("<external>",) + operand
            )
        operands = tuple(operands)
        relations.append((target, operands))

    own_ids = _prior_ids_in(member)
    assertions = []
    for prior_ids in context.assertions:
        inside = sorted(
            relative for relative, row in _all_rows(member) if row.prior_id in prior_ids
        )
        if not inside:
            continue
        assertions.append((tuple(inside), bool(prior_ids - own_ids)))

    member_paths = [other.path for other in members]
    external = set()
    for relative, row in _all_rows(member):
        if row.prior_id is None:
            continue
        for occurrence in context.direct_occurrences.get(row.prior_id, ()):
            if not any(occurrence[: len(path)] == path for path in member_paths):
                external.add(relative)
                break

    return (
        tuple(sorted(relations)),
        tuple(sorted(assertions)),
        tuple(sorted(external)),
    )


def _shared_split(members: List[ComponentNode]) -> Tuple[List[Tuple], Tuple[int, ...]]:
    """
    Rule R2: partition a candidate plate by the **cross-member** priors its
    members carry.

    A prior in *every* member does not discriminate (the MGE centre) and is
    dropped from every set; a prior shared only *inside* one member is not
    cross-member at all and never reaches the partition.
    """
    ids_per_member = [_prior_ids_in(member) for member in members]
    counts: Dict[int, int] = {}
    for ids in ids_per_member:
        for prior_id in ids:
            counts[prior_id] = counts.get(prior_id, 0) + 1

    universal = tuple(
        sorted(
            prior_id
            for prior_id, count in counts.items()
            if count == len(members) and count > 1
        )
    )
    keys = [
        tuple(
            sorted(
                prior_id
                for prior_id in ids
                if counts[prior_id] > 1 and prior_id not in universal
            )
        )
        for ids in ids_per_member
    ]
    return keys, universal


def _representative_key(member_paths: List[Path], position: int) -> Optional[str]:
    """
    The key ``find_groups`` puts in place of the member index -- ``"0 - 29"``,
    exactly as ``model.info`` prints it.
    """
    grouped = find_groups([(path, 0) for path in member_paths], limit=0)
    if len(grouped) != 1:  # pragma: no cover - defensive
        return None
    path = grouped[0][0]
    if position >= len(path):  # pragma: no cover - defensive
        return None
    return str(path[position])


def _varies_by_member(members: List[ComponentNode]) -> Tuple[str, ...]:
    """
    Dotted row paths, relative to a member, whose **fixed value** differs
    between members -- the MGE's per-Gaussian ``sigma``, the group model's
    per-galaxy ``mass.centre``.
    """
    per_member = [dict(_all_rows(member)) for member in members]
    varies: List[str] = []
    for relative, row in _all_rows(members[0]):
        if row.dimensionality == "tuple":
            values = [
                tuple(
                    component.value for component in rows.get(relative, row).components
                )
                for rows in per_member
            ]
        else:
            values = [rows.get(relative, row).value for rows in per_member]
        if any(value != values[0] for value in values[1:]):
            varies.append(relative)
    # A tuple row that varies already names the tuple; drop its slots.
    tuples = {
        relative
        for relative, row in _all_rows(members[0])
        if row.dimensionality == "tuple"
    }
    return tuple(
        relative
        for relative in varies
        if not any(relative.startswith(f"{name}.") for name in tuples)
    )


def _repeats(
    member: ComponentNode,
    shared_within: set,
    varies: Tuple[str, ...],
) -> Tuple[str, ...]:
    """
    One line naming what the plate repeats, built mechanically from the
    representative's own rows.
    """
    described = []
    for row in member.rows:
        name = row.name
        prior_ids = {row.prior_id} | {
            component.prior_id for component in row.components
        }
        if prior_ids & shared_within:
            described.append(f"{name} \u21c4 shared")
        elif row.sampling == "fixed" and name in varies:
            described.append(f"{name} fixed (varies by member)")
        else:
            described.append(f"{name} {row.sampling}")
    if not described:
        return (f"{member.cls_name}",)
    return (f"{member.cls_name} with priors " + ", ".join(described),)


def _plate(
    members: List[ComponentNode],
    representative: ComponentNode,
    context: _CollapseContext,
) -> ComponentNode:
    """
    One :class:`ComponentNode` standing for every member of a plate.

    ``members`` are the **uncollapsed** member subtrees -- the rules read the
    model as declared -- while ``representative`` is the first member as it
    comes out of the bottom-up pass, i.e. with its own children already
    collapsed.
    """
    member_paths = [member.path for member in members]
    position = len(representative.path) - 1

    _, universal = _shared_split(members)
    ids_per_member = [_prior_ids_in(member) for member in members]
    counts: Dict[int, int] = {}
    for ids in ids_per_member:
        for prior_id in ids:
            counts[prior_id] = counts.get(prior_id, 0) + 1
    shared_within = {prior_id for prior_id, count in counts.items() if count > 1}

    varies = _varies_by_member(members)
    return replace(
        representative,
        plate=PlateInfo(
            count=len(members),
            member_paths=tuple(member_paths),
            representative_key=_representative_key(member_paths, position),
            repeats=_repeats(members[0], shared_within, varies),
            shared_in_all=universal,
            varies_by_member=varies,
        ),
    )


def _collapse_siblings(
    node: ComponentNode,
    collapsed_children: Tuple[ComponentNode, ...],
    context: _CollapseContext,
) -> ComponentNode:
    """
    Rules R1 (soft plate) and R2 (shared split) plus their safety condition,
    applied to one node's ``children``.

    Only ``kind="model"`` siblings collapse: a ``Collection`` is a *frame*, not
    a repeated component, so it never collapses -- but its model children do.

    ``node`` is the **uncollapsed** node, so every rule reads the model as
    declared: after the bottom-up pass a member's own children may already be a
    plate, and a plate hides the very prior ids and rows R2 and the safety
    condition partition on.  ``collapsed_children`` are the same children after
    that pass, and are what is actually emitted.

    Everything is ordered by declaration, never by set iteration, so the result
    is deterministic.
    """
    children = node.children
    candidates = [
        index for index, child in enumerate(children) if child.kind == "model"
    ]
    if len(candidates) < 2:
        return replace(node, children=collapsed_children)

    by_signature: Dict[Tuple, List[int]] = {}
    for index in candidates:
        by_signature.setdefault(_signature(children[index], context), []).append(index)

    groups: List[List[int]] = []
    for indices in by_signature.values():
        if len(indices) < 2:
            groups.append(indices)
            continue
        members = [children[index] for index in indices]
        keys, _ = _shared_split(members)
        buckets: Dict[Tuple, List[int]] = {}
        for index, member, key in zip(indices, members, keys):
            buckets.setdefault(
                (key, _safety_profile(member, members, context)), []
            ).append(index)
        groups.extend(buckets.values())

    group_of: Dict[int, List[int]] = {}
    for indices in groups:
        for index in indices:
            group_of[index] = indices

    emitted: List[ComponentNode] = []
    for index in range(len(children)):
        indices = group_of.get(index)
        if indices is None or len(indices) == 1:
            emitted.append(collapsed_children[index])
            continue
        if index != indices[0]:
            continue
        emitted.append(
            _plate(
                [children[i] for i in indices],
                collapsed_children[indices[0]],
                context,
            )
        )

    return replace(node, children=tuple(emitted))


def _collapse_tree(node: ComponentNode, context: _CollapseContext) -> ComponentNode:
    """Apply :func:`_collapse_siblings` recursively, bottom-up."""
    return _collapse_siblings(
        node,
        tuple(_collapse_tree(child, context) for child in node.children),
        context,
    )


# ----------------------------------------------------------------------------
# the spec
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphSpec:
    """
    The semantic layer: a nesting tree plus the three edge kinds, a path index
    and the reconciling counts.
    """

    root: ComponentNode
    shared: Tuple[SharedEdge, ...] = ()
    relations: Tuple[RelationEdge, ...] = ()
    assertions: Tuple[AssertionEdge, ...] = ()
    #: element key -> the ``model.info`` paths it resolves to.  See
    #: :func:`_path_index` for the two value shapes.
    path_index: Dict[str, Any] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_model(
        cls,
        model,
        analysis=None,
        collapse: bool = True,
        solved_paths: Sequence = (),
    ) -> "GraphSpec":
        """
        Extract the spec of a model.

        Parameters
        ----------
        model
            Any ``AbstractPriorModel`` -- ``af.Model``, ``af.Collection``,
            ``af.Array`` or a ``FactorGraphModel``'s ``global_prior_model``.
        analysis
            An optional ``af.Analysis``.  When given, its latent catalogue
            (``type(analysis).Latent.keys(analysis)``) is attached as ``solved``
            rows with ``in_model_info=False``.  Without it, latents are simply
            absent.
        collapse
            Whether to apply rules R1/R2 and their safety condition, collapsing
            repeated sibling components into plates (see
            :func:`_collapse_siblings`).  ``collapse=False`` returns the
            uncollapsed tree.
        solved_paths
            Paths -- tuples or dotted strings -- whose rows are re-stated as
            ``solved`` and marked absent from ``model.info``.  Phase 3 supplies
            the domain rules that populate this.
        """
        extractor = _Extractor(model, analysis=analysis, solved_paths=solved_paths)
        raw_root = extractor.attach_latents(extractor.build_root())
        root = (
            _collapse_tree(raw_root, _CollapseContext(extractor))
            if collapse
            else raw_root
        )
        spec = cls(
            root=root,
            shared=extractor.shared_edges(),
            relations=extractor.relation_edges(),
            assertions=extractor.assertion_edges(),
        )
        object.__setattr__(spec, "path_index", _path_index(spec, model))
        object.__setattr__(spec, "counts", _counts(model, spec, raw_root))
        return spec

    # -- convenience --------------------------------------------------------

    def components(self) -> List[ComponentNode]:
        """Every component node, in visual order."""

        def _walk(node):
            yield node
            for child in node.children:
                yield from _walk(child)

        return list(_walk(self.root))

    def node(self, path) -> Optional[ComponentNode]:
        """The component node at ``path``, or ``None``."""
        path = _as_path(path)
        for node in self.components():
            if node.path == path:
                return node
        return None

    def rows(self) -> List[ParamRow]:
        """Every top-level row of every component, in visual order."""
        return [row for node in self.components() for row in node.rows]

    def row(self, path) -> Optional[ParamRow]:
        """The row at ``path`` (top-level rows and tuple components), or ``None``."""
        path = _as_path(path)
        for row in self.rows():
            if row.path == path:
                return row
            for component in row.components:
                if component.path == path:
                    return component
        return None

    def to_dict(self) -> dict:
        """
        An insertion-ordered, JSON-serialisable dict.  ``json.dumps`` of it is
        byte-stable across extractions of the same model.
        """
        return {
            "root": self.root.to_dict(),
            "shared": [edge.to_dict() for edge in self.shared],
            "relations": [edge.to_dict() for edge in self.relations],
            "assertions": [edge.to_dict() for edge in self.assertions],
            "path_index": {
                key: (
                    entry if isinstance(entry, dict) else [list(path) for path in entry]
                )
                for key, entry in self.path_index.items()
            },
            "counts": dict(self.counts),
        }


def _info_group_map(model) -> Dict[Path, Path]:
    """
    ``concrete leaf path -> the grouped path`` ``model.info`` prints for it.

    A faithful replay of ``AbstractPriorModel.info``'s own
    ``find_groups(..., limit=1)`` pass, but carrying each group's *members*
    along so the mapping can be read back.  (``info`` additionally honours a
    parent's ``__exclude_identifier_fields__``; those attributes are skipped
    from the spec too, so they never reach this map.)
    """
    entries: List[Tuple[Path, Any, List[Path]]] = []
    for path, value in model.path_instance_tuples_for_class(
        (Prior, float, Constant, int, tuple, ConfigException), ignore_children=True
    ):
        if path[-1] in ("id", "item_number"):
            continue
        concrete = _as_path(path)
        entries.append((concrete, value, [concrete]))

    if not entries:
        return {}

    longest = max(len(path) for path, _, _ in entries)
    for position in range(longest - 1):
        grouped: Dict[Any, List] = {}
        carried: List[Tuple[Path, Any, List[Path]]] = []
        for path, value, sources in entries:
            if position >= len(path):
                carried.append((path, value, sources))
                continue
            key = (path[:position], path[position + 1 :], value)
            try:
                bucket = grouped.setdefault(key, [[], []])
            except TypeError:  # pragma: no cover - an unhashable leaf value
                carried.append((path, value, sources))
                continue
            bucket[0].append(path[position])
            bucket[1].extend(sources)
        for (before, after, value), (names, sources) in grouped.items():
            try:
                key = integers_representative_key(list(map(int, names)))
            except ValueError:
                key = (
                    f"{min(names)} - {max(names)}" if len(set(names)) > 1 else names[0]
                )
            carried.append(((*before, key, *after), value, sources))
        entries = carried

    return {source: path for path, _, sources in entries for source in sources}


def _group_paths(paths: Sequence[Path]) -> Tuple[Path, ...]:
    """The figure's own grouping of a plate's member paths, via ``find_groups``."""
    if len(paths) < 2:
        return tuple(paths)
    return tuple(
        path for path, _ in find_groups([(path, 0) for path in paths], limit=0)
    )


def _path_index(spec: GraphSpec, model) -> Dict[str, Any]:
    """
    element key -> the ``model.info`` paths it resolves to.

    Two value shapes, both JSON-stable:

    * a **tuple of paths** for an element standing for exactly one path -- the
      ordinary case, and the phase-1 shape.  An element absent from
      ``model.info`` (a ``solved`` row, a ``latent`` row, an assertion) is the
      empty tuple; a ``missing`` row *is* in ``model.info`` and keeps its path.
    * a ``{"figure": [...], "info": [...]}`` **dict** of ``"/"``-joined paths
      for an element inside a plate whose figure partition is *finer* than
      ``model.info``'s grouping -- the MGE's two 30-member plates against
      ``model.info``'s single ``0 - 59`` centre.  ``figure`` is the plate's own
      grouping of the member paths, ``info`` is what ``model.info`` prints.  The
      mapping is recorded rather than hidden (epic: "record the mapping rather
      than hiding it").  When the two agree the plain tuple shape is used.
    """
    index: Dict[str, Any] = {}
    info_map = _info_group_map(model) if _has_plate(spec.root) else {}

    def _entry(concrete: Tuple[Path, ...]) -> Any:
        figure = _group_paths(concrete)
        if len(concrete) < 2:
            return figure
        info: List[Path] = []
        for path in concrete:
            grouped = info_map.get(path, path)
            if grouped not in info:
                info.append(grouped)
        if tuple(info) == figure:
            return figure
        return {
            "figure": [_key(path) for path in figure],
            "info": [_key(path) for path in info],
        }

    def _add_row(row: ParamRow, concrete: Tuple[Path, ...]):
        index[_key(row.path)] = _entry(concrete) if row.in_model_info else ()
        for component in row.components:
            leaf = component.path[len(row.path) :]
            if not component.in_model_info:
                index[_key(component.path)] = ()
            elif row.dimensionality == "tuple" and row.prior_cls_name == "tuple":
                # A fixed tuple constant is one grouped line in `model.info`.
                index[_key(component.path)] = index[_key(row.path)]
            else:
                index[_key(component.path)] = _entry(
                    tuple(path + leaf for path in concrete)
                )

    def _walk(node: ComponentNode, concrete: Tuple[Path, ...]):
        index[_key(node.path)] = _entry(concrete)
        for row in node.rows:
            suffix = row.path[len(node.path) :]
            _add_row(row, tuple(path + suffix for path in concrete))
        for child in node.children:
            if child.plate is not None:
                suffixes = [
                    member[len(node.path) :] for member in child.plate.member_paths
                ]
            else:
                suffixes = [child.path[len(node.path) :]]
            _walk(
                child,
                tuple(path + suffix for path in concrete for suffix in suffixes),
            )

    _walk(spec.root, (spec.root.path,))
    for number, _ in enumerate(spec.assertions):
        index[f"assertion/{number}"] = ()
    return index


def _has_plate(node: ComponentNode) -> bool:
    return node.plate is not None or any(_has_plate(child) for child in node.children)


def _node_count(node: ComponentNode) -> int:
    return 1 + sum(_node_count(child) for child in node.children)


def _plate_count(node: ComponentNode) -> int:
    return (node.plate is not None) + sum(
        _plate_count(child) for child in node.children
    )


def _counts(model, spec: GraphSpec, raw_root: ComponentNode) -> Dict[str, int]:
    """
    The reconciling counts.

    The row-derived counts (``fixed_leaf_slots``, ``missing``) are of the
    **uncollapsed** tree: a plate stands for every one of its members, so
    collapsing must not make a fixed value look absent from the model.  Only the
    component counts are of both trees -- ``components`` after collapse,
    ``components_raw`` before, and ``plates`` how many of the former stand for
    more than one of the latter.
    """
    fixed_leaf_slots = 0
    missing = 0
    for node in _walk_nodes(raw_root):
        for row in node.rows:
            if row.dimensionality == "tuple":
                for component in row.components:
                    if component.sampling == "fixed":
                        fixed_leaf_slots += 1
                    if component.sampling == "missing":
                        missing += 1
                continue
            if row.sampling == "fixed":
                fixed_leaf_slots += 1
            if row.sampling == "missing":
                missing += 1
    return {
        "unique_sampled_scalars": model.prior_count,
        "fixed_leaf_slots": fixed_leaf_slots,
        "shared_priors": len(spec.shared),
        "missing": missing,
        "components_raw": _node_count(raw_root),
        "components": _node_count(spec.root),
        "plates": _plate_count(spec.root),
    }


def _walk_nodes(node: ComponentNode):
    yield node
    for child in node.children:
        yield from _walk_nodes(child)


def graph_spec_from(model, **kwargs) -> GraphSpec:
    """Module-level alias for :meth:`GraphSpec.from_model`."""
    return GraphSpec.from_model(model, **kwargs)
