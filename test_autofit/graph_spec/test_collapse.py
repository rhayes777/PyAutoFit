"""
The collapse rules -- R1 (soft plate), R2 (shared split) and the mandatory
safety condition -- from the phase-1 prompt
(``PyAutoMind/active/model_figures_1_graph_spec.md``, "Collapse rules R1/R2,
with the safety condition").

Every assertion is on the extracted :class:`~autofit.graph_spec.GraphSpec`.
"""

import json

import autofit as af
from autofit.graph_spec import GraphSpec

# `af.ex.Gaussian`'s configured priors, repeated verbatim so a prior assigned
# by hand is *configuration-identical* to the default one and rule R1 cannot
# tell them apart.
CONFIGURED_SIGMA = dict(lower_limit=0.0, upper_limit=25.0)
CONFIGURED_CENTRE = dict(lower_limit=0.0, upper_limit=100.0)


class Scale:
    """A one-parameter component that is *not* a Gaussian, so it never joins."""

    def __init__(self, factor=1.0):
        self.factor = factor


class Blob:
    """A component with a tuple parameter and an ``af.Model(int)`` slot."""

    def __init__(self, centre=(0.0, 0.0), intensity=1.0, pixels=1):
        self.centre = centre
        self.intensity = intensity
        self.pixels = pixels


class Pair:
    """Two sub-components, so a prior can be shared *inside* one member."""

    def __init__(self, first=None, second=None):
        self.first = first
        self.second = second


def _gaussians(count):
    return [af.Model(af.ex.Gaussian) for _ in range(count)]


def _plates(spec):
    return [node for node in spec.components() if node.plate is not None]


# ------------------------------------------------------------------ R1


def test_r1_collapses_identical_siblings_into_one_plate():
    spec = GraphSpec.from_model(af.Collection(_gaussians(3)))

    (plate,) = _plates(spec)
    assert plate.plate.count == 3
    assert plate.plate.member_paths == (("0",), ("1",), ("2",))
    # The key `model.info` itself prints for the group, via `find_groups`.
    assert plate.plate.representative_key == "0 - 2"
    assert plate.plate.repeats == (
        "Gaussian with priors centre free, normalization free, sigma free",
    )

    # The representative is the first member, rows and all.
    assert [child.path for child in spec.root.children] == [("0",)]
    assert [row.name for row in plate.rows] == ["centre", "normalization", "sigma"]

    assert spec.counts["components_raw"] == 4
    assert spec.counts["components"] == 2
    assert spec.counts["plates"] == 1
    # Collapsing never changes what the model actually samples.
    assert spec.counts["unique_sampled_scalars"] == 9


def test_r1_a_different_prior_configuration_stays_out_of_the_plate():
    first, odd, third, fourth = _gaussians(4)
    odd.centre = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)
    spec = GraphSpec.from_model(
        af.Collection([first, odd, third, fourth])
    )

    # Declaration order is preserved: the plate takes its first member's slot.
    assert [child.path for child in spec.root.children] == [("0",), ("1",)]
    plate, loner = spec.root.children
    assert plate.plate.count == 3
    assert plate.plate.member_paths == (("0",), ("2",), ("3",))
    assert plate.plate.representative_key == "0, 2 - 3"
    assert loner.plate is None
    assert loner.path == ("1",)


def test_r1_ignores_prior_label_and_constant_values():
    models = _gaussians(3)
    for index, model in enumerate(models):
        # A hand-assigned prior picks up its own `_label` counter; R1 must not
        # see it, and the configuration is identical to the default.
        model.normalization = af.LogUniformPrior(
            lower_limit=1e-06, upper_limit=1000000.0
        )
        model.sigma = float(index + 1)

    labels = {model.normalization._label for model in models}
    assert len(labels) == 3

    spec = GraphSpec.from_model(af.Collection(models))
    (plate,) = _plates(spec)

    assert plate.plate.count == 3
    # The differing constants are *named*, never silently merged away.
    assert plate.plate.varies_by_member == ("sigma",)
    assert "sigma fixed (varies by member)" in plate.plate.repeats[0]
    assert plate.rows[-1].value == 1.0
    # Every member's fixed slot is still counted.
    assert spec.counts["fixed_leaf_slots"] == 3


# ------------------------------------------------------------------ R2


def test_r2_cross_member_sharing_splits_the_plate():
    models = _gaussians(4)
    centre = af.UniformPrior(**CONFIGURED_CENTRE)
    left = af.UniformPrior(**CONFIGURED_SIGMA)
    right = af.UniformPrior(**CONFIGURED_SIGMA)
    for model in models:
        model.centre = centre
    models[0].sigma = left
    models[1].sigma = left
    models[2].sigma = right
    models[3].sigma = right

    spec = GraphSpec.from_model(af.Collection(models))
    plates = _plates(spec)

    assert [plate.plate.count for plate in plates] == [2, 2]
    assert [plate.plate.representative_key for plate in plates] == ["0 - 1", "2 - 3"]
    assert [plate.plate.member_paths for plate in plates] == [
        (("0",), ("1",)),
        (("2",), ("3",)),
    ]
    # The centre is in *every* member of the candidate plate, so it does not
    # discriminate and never splits it; what splits is `left` against `right`.
    # `shared_in_all` is then reported for each *resulting* plate, where the
    # discriminating prior is itself carried by every member.
    assert plates[0].plate.shared_in_all == tuple(sorted((centre.id, left.id)))
    assert plates[1].plate.shared_in_all == tuple(sorted((centre.id, right.id)))


def test_r2_a_prior_in_every_member_does_not_split():
    models = _gaussians(4)
    centre = af.UniformPrior(**CONFIGURED_CENTRE)
    for model in models:
        model.centre = centre

    spec = GraphSpec.from_model(af.Collection(models))
    (plate,) = _plates(spec)

    assert plate.plate.count == 4
    assert plate.plate.shared_in_all == (centre.id,)
    assert "centre ⇄ shared" in plate.plate.repeats[0]


def test_r2_a_prior_shared_only_inside_one_member_does_not_split():
    pairs = [
        af.Model(
            Pair,
            first=af.Model(af.ex.Gaussian),
            second=af.Model(af.ex.Gaussian),
        )
        for _ in range(3)
    ]
    # Both occurrences are inside member 0, so this is not cross-member sharing.
    pairs[0].first.centre = pairs[0].second.centre

    spec = GraphSpec.from_model(af.Collection(pairs))
    outer = [node for node in _plates(spec) if node.cls_name == "Pair"]

    assert len(outer) == 1
    assert outer[0].plate.count == 3
    assert outer[0].plate.shared_in_all == ()


# ------------------------------------------------------------------ safety


def test_safety_a_member_with_a_relation_leaves_the_plate():
    models = _gaussians(3)
    models[1].centre = models[1].normalization + models[1].sigma

    spec = GraphSpec.from_model(af.Collection(models))

    assert [child.path for child in spec.root.children] == [("0",), ("1",)]
    plate, relation_member = spec.root.children
    assert plate.plate.count == 2
    assert plate.plate.member_paths == (("0",), ("2",))
    assert relation_member.plate is None
    assert relation_member.rows[0].provenance.kind == "relation"


def test_safety_relations_that_differ_only_in_operand_location_split():
    """
    The safety condition, not R1: both members carry a ``MultiplePrior`` at the
    same relative path with the same configuration, so their *rows* are
    indistinguishable.  What differs is where the operand lives -- inside the
    member for one, outside the plate for the other -- and a plate must preserve
    relations across its members.
    """
    inside, outside, plain = _gaussians(3)
    scale = af.Model(Scale, factor=af.UniformPrior(**CONFIGURED_SIGMA))

    inside.centre = inside.sigma * 2.0
    outside.centre = scale.factor * 2.0

    spec = GraphSpec.from_model(
        af.Collection(inside=inside, outside=outside, plain=plain, scale=scale)
    )

    assert [child.path for child in spec.root.children] == [
        ("inside",),
        ("outside",),
        ("plain",),
        ("scale",),
    ]
    assert all(child.plate is None for child in spec.root.children)


def test_safety_a_member_touched_by_an_assertion_leaves_the_plate():
    models = _gaussians(3)
    model = af.Collection(models)
    model.add_assertion(models[0].sigma > 0.5)

    spec = GraphSpec.from_model(model)

    assert len(spec.assertions) == 1
    assert [child.path for child in spec.root.children] == [("0",), ("1",)]
    asserted, plate = spec.root.children
    assert asserted.plate is None
    assert plate.plate.count == 2
    assert plate.plate.member_paths == (("1",), ("2",))


def test_safety_a_member_sharing_outside_the_plate_leaves_it():
    models = _gaussians(3)
    scale = af.Model(Scale, factor=af.UniformPrior(**CONFIGURED_SIGMA))
    model = af.Collection(a=models[0], b=models[1], c=models[2], scale=scale)
    model.a.sigma = model.scale.factor

    spec = GraphSpec.from_model(model)

    assert [child.path for child in spec.root.children] == [
        ("a",),
        ("b",),
        ("scale",),
    ]
    leaver, plate, _ = spec.root.children
    assert leaver.plate is None
    assert plate.plate.count == 2
    assert plate.plate.member_paths == (("b",), ("c",))
    assert spec.counts["shared_priors"] == 1


# ------------------------------------------------------------------ traps


def test_fixed_tuple_constants_and_model_int_survive_a_plate():
    blobs = []
    for index in range(3):
        blob = af.Model(
            Blob,
            centre=(0.1 * index, 0.2 * index),
            intensity=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
            pixels=af.Model(int),
        )
        blobs.append(blob)

    spec = GraphSpec.from_model(af.Collection(blobs))
    (plate,) = _plates(spec)

    assert plate.plate.count == 3
    assert plate.plate.varies_by_member == ("centre",)

    centre = spec.row(("0", "centre"))
    assert centre.dimensionality == "tuple"
    assert centre.sampling == "fixed"
    assert [component.value for component in centre.components] == [0.0, 0.0]
    assert [component.name for component in centre.components] == [
        "centre_0",
        "centre_1",
    ]

    # Rule R7: `af.Model(int)` is a row on its owner, never a component.
    assert spec.node(("0", "pixels")) is None
    assert spec.row(("0", "pixels")) is not None
    assert [row.name for row in plate.rows] == ["centre", "intensity", "pixels"]

    # Nothing fixed is dropped: three members x (two tuple slots + `pixels`).
    assert spec.counts["fixed_leaf_slots"] == 9


def test_collections_are_frames_and_never_collapse():
    spec = GraphSpec.from_model(
        af.Collection(
            a=af.Collection(g=af.Model(af.ex.Gaussian)),
            b=af.Collection(g=af.Model(af.ex.Gaussian)),
        )
    )

    assert [child.path for child in spec.root.children] == [("a",), ("b",)]
    assert all(child.plate is None for child in spec.root.children)
    # ... but the model children *inside* a collection still do.
    assert spec.counts["plates"] == 0
    assert spec.counts["components"] == spec.counts["components_raw"] == 5


def test_a_collections_model_children_do_collapse():
    spec = GraphSpec.from_model(
        af.Collection(outer=af.Collection(_gaussians(4)))
    )
    (plate,) = _plates(spec)

    assert plate.path == ("outer", "0")
    assert plate.plate.count == 4
    assert plate.plate.representative_key == "0 - 3"


# ------------------------------------------------------------------ contract


def test_collapse_false_leaves_everything_expanded():
    model = af.Collection(_gaussians(3))
    spec = GraphSpec.from_model(model, collapse=False)

    assert [child.path for child in spec.root.children] == [("0",), ("1",), ("2",)]
    assert all(node.plate is None for node in spec.components())
    assert spec.counts["components"] == spec.counts["components_raw"] == 4
    assert spec.counts["plates"] == 0


def test_the_path_index_records_both_partitions_for_a_plate():
    models = _gaussians(4)
    centre = af.UniformPrior(**CONFIGURED_CENTRE)
    left = af.UniformPrior(**CONFIGURED_SIGMA)
    right = af.UniformPrior(**CONFIGURED_SIGMA)
    for model in models:
        model.centre = centre
    models[0].sigma = models[1].sigma = left
    models[2].sigma = models[3].sigma = right

    spec = GraphSpec.from_model(af.Collection(models))

    # `model.info` groups the shared centre across all four; the figure's
    # partition is finer (two plates), so both are recorded.
    assert spec.path_index["0/centre"] == {
        "figure": ["0 - 1/centre"],
        "info": ["0 - 3/centre"],
    }
    # Where the two agree -- `sigma` is shared by exactly this plate's members,
    # so `model.info` groups it the same way -- the plain tuple-of-paths shape
    # is kept.
    assert spec.path_index["0/sigma"] == (("0 - 1", "sigma"),)
    # A row that resolves to exactly one path keeps the phase-1 shape too.
    assert spec.path_index["0/normalization"] == {
        "figure": ["0 - 1/normalization"],
        "info": ["0/normalization", "1/normalization"],
    }


def test_collapse_is_deterministic():
    def build():
        models = _gaussians(6)
        centre = af.UniformPrior(**CONFIGURED_CENTRE)
        for model in models:
            model.centre = centre
        left = af.UniformPrior(**CONFIGURED_SIGMA)
        right = af.UniformPrior(**CONFIGURED_SIGMA)
        for model in models[:3]:
            model.sigma = left
        for model in models[3:]:
            model.sigma = right
        return af.Collection(models)

    model = build()
    first = json.dumps(GraphSpec.from_model(model).to_dict())
    second = json.dumps(GraphSpec.from_model(model).to_dict())

    assert first == second
    assert [plate.plate.count for plate in _plates(GraphSpec.from_model(model))] == [
        3,
        3,
    ]
