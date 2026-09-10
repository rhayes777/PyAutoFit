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
    same ``Prior`` object appears.  :attr:`ParamRow.shared` is a derived
    property.  **Sharing is never a sampling state.**
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
both ``('normalization',)`` and ``('centre', 'self')``.  Those extra occurrences
are therefore reported as sharing, exactly as ``model.info`` reports them.  This
is intentional -- the paths are genuine -- and the relation is *additionally*
carried as a :class:`RelationEdge`.

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
and the module-level ``graph_spec_from(model, **kwargs)``.  ``collapse`` is
accepted **and ignored** in phase 1 -- :func:`_collapse_siblings` is the named
hook the plate/collapse phase fills in.
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
    dimensionality: str = "scalar"
    components: Tuple["ParamRow", ...] = ()
    provenance: Provenance = field(default_factory=lambda: Provenance("config-default"))
    value: Any = None
    prior_cls_name: Optional[str] = None
    in_model_info: bool = True
    is_instance: bool = False

    @property
    def shared(self) -> bool:
        """Whether this row's prior object appears at more than one path."""
        return len(self.occurrences) > 1

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": list(self.path),
            "sampling": self.sampling,
            "prior_id": self.prior_id,
            "occurrences": [list(path) for path in self.occurrences],
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

    Defined here so the spec's shape is fixed in phase 1; **the collapse phase
    fills it in**.  ``ComponentNode.plate`` is ``None`` for every node this
    module produces.

    Parameters
    ----------
    count
        How many components the plate stands for.
    member_paths
        The path of every member, in declaration order.
    representative_key
        A stable key for the member drawn as the representative.
    repeats
        What the plate repeats -- the safety condition of rules R1/R2 requires
        the plate to preserve sharing, relations, assertions and exceptions
        across its members, and this field records which of those it asserts.
    shared_in_all
        Parameter names whose prior is shared by *every* member (these do not
        discriminate, so they never split a plate).
    varies_by_member
        Parameter names whose fixed value differs between members.
    """

    count: int
    member_paths: Tuple[Path, ...] = ()
    representative_key: Optional[str] = None
    repeats: Tuple[str, ...] = ()
    shared_in_all: Tuple[str, ...] = ()
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

        self.occurrences: Dict[int, Tuple[Path, ...]] = {}
        for paths, prior in model.all_paths_prior_tuples:
            self.occurrences[prior.id] = tuple(_as_path(path) for path in paths)

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

        return _collapse_siblings(
            ComponentNode(
                path=path,
                name=name,
                cls_name=_cls_name(obj),
                kind=_kind(obj),
                obj_id=getattr(obj, "id", None),
                rows=tuple(rows),
                children=tuple(children),
            )
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
        return tuple(
            SharedEdge(prior.id, tuple(_as_path(path) for path in paths))
            for paths, prior in self.model.all_paths_prior_tuples
            if len(paths) > 1
        )

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


def _collapse_siblings(node: ComponentNode) -> ComponentNode:
    """
    The collapse hook -- rules R1 (soft plate) and R2 (shared split) plus their
    safety condition land here, filling :class:`PlateInfo`.

    **TODO (collapse phase):** phase 1 performs no collapse; this returns its
    input unchanged and ``GraphSpec.from_model(collapse=...)`` is accepted and
    ignored.
    """
    return node


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
    path_index: Dict[str, Tuple[Path, ...]] = field(default_factory=dict)
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
            **Accepted and ignored in phase 1** (see :func:`_collapse_siblings`).
        solved_paths
            Paths -- tuples or dotted strings -- whose rows are re-stated as
            ``solved`` and marked absent from ``model.info``.  Phase 3 supplies
            the domain rules that populate this.
        """
        extractor = _Extractor(model, analysis=analysis, solved_paths=solved_paths)
        root = extractor.build_root()
        root = extractor.attach_latents(root)
        spec = cls(
            root=root,
            shared=extractor.shared_edges(),
            relations=extractor.relation_edges(),
            assertions=extractor.assertion_edges(),
        )
        object.__setattr__(spec, "path_index", _path_index(spec))
        object.__setattr__(spec, "counts", _counts(model, spec))
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
                key: [list(path) for path in paths]
                for key, paths in self.path_index.items()
            },
            "counts": dict(self.counts),
        }


def _path_index(spec: GraphSpec) -> Dict[str, Tuple[Path, ...]]:
    """
    element key -> the ``model.info`` paths it resolves to.

    An element absent from ``model.info`` (``solved`` and ``latent`` rows, every
    assertion) is recorded with an **empty** tuple; a ``missing`` row is in
    ``model.info`` and keeps its path.
    """
    index: Dict[str, Tuple[Path, ...]] = {}

    def _add_row(row: ParamRow):
        index[_key(row.path)] = (row.path,) if row.in_model_info else ()
        for component in row.components:
            if not component.in_model_info:
                index[_key(component.path)] = ()
            elif row.dimensionality == "tuple" and row.prior_cls_name == "tuple":
                # A fixed tuple constant is one grouped line in `model.info`.
                index[_key(component.path)] = (row.path,)
            else:
                index[_key(component.path)] = (component.path,)

    def _walk(node: ComponentNode):
        index[_key(node.path)] = (node.path,)
        for row in node.rows:
            _add_row(row)
        for child in node.children:
            _walk(child)

    _walk(spec.root)
    for number, _ in enumerate(spec.assertions):
        index[f"assertion/{number}"] = ()
    return index


def _counts(model, spec: GraphSpec) -> Dict[str, int]:
    fixed_leaf_slots = 0
    missing = 0
    for row in spec.rows():
        if row.dimensionality == "tuple":
            for component in row.components:
                if component.sampling == "fixed":
                    fixed_leaf_slots += 1
                if component.sampling == "missing":
                    missing += 1
            if row.sampling == "missing":
                # counted per slot above
                pass
            continue
        if row.sampling == "fixed":
            fixed_leaf_slots += 1
        if row.sampling == "missing":
            missing += 1
    components_raw = len(spec.components())
    return {
        "unique_sampled_scalars": model.prior_count,
        "fixed_leaf_slots": fixed_leaf_slots,
        "shared_priors": len(spec.shared),
        "missing": missing,
        "components_raw": components_raw,
        "components": components_raw,
    }


def graph_spec_from(model, **kwargs) -> GraphSpec:
    """Module-level alias for :meth:`GraphSpec.from_model`."""
    return GraphSpec.from_model(model, **kwargs)
