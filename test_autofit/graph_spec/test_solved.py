"""
The ``solved`` semantics: the ``__solved_parameters__`` protocol, the additive
``solved_paths=`` and rule R7's float subclasses.

PyAutoFit must never know a lens class name, so a *class* declares the
quantities a fit solves for it -- a linear light profile's ``intensity``, a
``PointSolved``'s ``centre``, a pixelization's ``reconstruction`` -- and this
module reads that declaration off ``obj.cls`` exactly as it reads
``__exclude_identifier_fields__`` off the type.  The doubles here carry the same
shapes as the real profiles (``test_autofit/graph_spec/lens_doubles.py``:
autofit imports autonerves only, never autogalaxy or autolens).
"""

import itertools
import json

import autofit as af
from autofit.graph_spec import GraphSpec, graph_spec_from

from .lens_doubles import Galaxy, sersic


class LinearGaussian:
    """
    A linear light profile: ``intensity`` is solved by the inversion and is
    absent from the model entirely, so the class declares it.
    """

    __solved_parameters__ = ("intensity",)

    def __init__(self, centre=(0.0, 0.0), sigma=1.0):
        self.centre = centre
        self.sigma = sigma


class LinearSersic(LinearGaussian):
    """A subclass inherits the declaration, as ``lp_linear.Sersic`` does."""

    def __init__(self, centre=(0.0, 0.0), sigma=1.0, sersic_index=4.0):
        super().__init__(centre=centre, sigma=sigma)
        self.sersic_index = sersic_index


class PointSolved:
    """Zero parameters: the centre is solved analytically."""

    __solved_parameters__ = ("centre",)


class Reexposed:
    """A class that declares a name which *is* one of its own slots."""

    __solved_parameters__ = ("sigma",)

    def __init__(self, centre=(0.0, 0.0), sigma=1.0):
        self.centre = centre
        self.sigma = sigma


class Malformed:
    """A declaration that is not a tuple of ``str`` is ignored, never obeyed."""

    __solved_parameters__ = "intensity"

    def __init__(self, sigma=1.0):
        self.sigma = sigma


class Redshift(float):
    """
    The shape of autogalaxy's ``Redshift`` -- a thin ``float`` subclass whose
    ``__new__`` takes the value, so ``af.Model(Redshift)`` is one scalar.
    """

    def __new__(cls, redshift):
        return float.__new__(cls, redshift)

    def __init__(self, redshift):
        float.__init__(redshift)


def linear_gaussian(sigma=1.0, cls=LinearGaussian):
    model = af.Model(cls)
    model.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    model.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
    model.sigma = sigma
    if cls is LinearSersic:
        model.sersic_index = af.UniformPrior(lower_limit=0.8, upper_limit=8.0)
    return model


# ---------------------------------------------------------------- the protocol


def test_a_declared_solved_parameter_is_appended_as_a_row():
    spec = GraphSpec.from_model(af.Collection(bulge=linear_gaussian()))
    node = spec.node(("bulge",))

    # Appended after the model's own slots, in declaration order.
    assert [row.name for row in node.rows] == ["centre", "sigma", "intensity"]

    row = spec.row(("bulge", "intensity"))
    assert row.sampling == "solved"
    assert row.provenance.kind == "solved-by-fit"
    # Solved quantities have no counterpart in `model.info` at all.
    assert row.in_model_info is False
    assert spec.path_index["bulge/intensity"] == ()
    assert "intensity" not in af.Collection(bulge=linear_gaussian()).info
    assert spec.counts["solved"] == 1
    # The model's own counts are untouched: nothing was added to the model.
    assert spec.counts["unique_sampled_scalars"] == 2


def test_a_subclass_inherits_the_declaration():
    spec = GraphSpec.from_model(af.Collection(bulge=linear_gaussian(cls=LinearSersic)))

    assert [row.name for row in spec.node(("bulge",)).rows] == [
        "centre",
        "sigma",
        "sersic_index",
        "intensity",
    ]
    assert spec.row(("bulge", "intensity")).sampling == "solved"
    assert spec.counts["solved"] == 1


def test_a_class_with_no_slots_at_all_is_a_card_of_one_solved_row():
    """``PointSolved`` has zero parameters -- without the declaration the box
    would read "no parameters", which misreports an analytically solved centre."""
    spec = GraphSpec.from_model(af.Collection(point=af.Model(PointSolved)))
    (row,) = spec.node(("point",)).rows

    assert (row.name, row.sampling) == ("centre", "solved")
    assert row.provenance.kind == "solved-by-fit"
    assert row.in_model_info is False
    assert spec.counts["solved"] == 1
    assert spec.counts["unique_sampled_scalars"] == 0


def test_a_declared_name_that_is_a_real_slot_is_retagged_not_duplicated():
    profile = af.Model(Reexposed)
    profile.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=5.0)
    spec = GraphSpec.from_model(af.Collection(profile=profile))
    rows = spec.node(("profile",)).rows

    assert [row.name for row in rows] == ["centre", "sigma"]
    row = spec.row(("profile", "sigma"))
    assert row.sampling == "solved"
    assert row.in_model_info is False
    # Re-tagged in place: it is still the model's own prior, not a synthesised
    # row, so its provenance and prior identity survive.
    assert row.provenance.kind == "config-default"
    assert row.prior_id is not None
    assert spec.path_index["profile/sigma"] == ()
    assert spec.counts["solved"] == 1


def test_a_malformed_declaration_is_ignored():
    spec = GraphSpec.from_model(af.Collection(profile=af.Model(Malformed)))

    assert [row.name for row in spec.node(("profile",)).rows] == ["sigma"]
    assert spec.counts["solved"] == 0


def test_a_declared_name_that_is_a_child_component_is_not_shadowed():
    """A component of its own is never replaced by a solved row."""

    class Owner:
        __solved_parameters__ = ("bulge",)

        def __init__(self, bulge=None):
            self.bulge = bulge

    spec = GraphSpec.from_model(af.Model(Owner, bulge=sersic()))

    assert spec.root.rows == ()
    assert [child.name for child in spec.root.children] == ["bulge"]
    assert spec.counts["solved"] == 0


# ---------------------------------------------------------------- solved_paths


def test_an_unmatched_solved_path_is_synthesised_on_its_owner():
    model = af.Collection(gaussian=af.Model(af.ex.Gaussian))
    spec = GraphSpec.from_model(model, solved_paths=("gaussian.intensity",))

    assert [row.name for row in spec.node(("gaussian",)).rows] == [
        "centre",
        "normalization",
        "sigma",
        "intensity",
    ]
    row = spec.row(("gaussian", "intensity"))
    assert row.sampling == "solved"
    assert row.provenance.kind == "solved-by-fit"
    assert row.in_model_info is False
    assert spec.path_index["gaussian/intensity"] == ()
    assert spec.counts["solved"] == 1


def test_an_unmatched_solved_path_whose_owner_is_not_a_component_lands_on_root():
    """Never silently ignored: the row goes on the root rather than nowhere."""
    model = af.Collection(gaussian=af.Model(af.ex.Gaussian))
    spec = GraphSpec.from_model(model, solved_paths=("nowhere.thing",))

    (row,) = spec.root.rows
    assert (row.name, row.path) == ("thing", ("nowhere", "thing"))
    assert row.sampling == "solved"
    assert row.provenance.kind == "solved-by-fit"
    assert spec.path_index["nowhere/thing"] == ()
    assert spec.counts["solved"] == 1


def test_a_matched_solved_path_is_still_retagged_in_place():
    model = af.Collection(gaussian=af.Model(af.ex.Gaussian))
    spec = GraphSpec.from_model(model, solved_paths=("gaussian.normalization",))

    assert [row.name for row in spec.node(("gaussian",)).rows] == [
        "centre",
        "normalization",
        "sigma",
    ]
    row = spec.row(("gaussian", "normalization"))
    assert row.sampling == "solved"
    assert row.in_model_info is False
    assert spec.counts["solved"] == 1


def test_solved_paths_and_the_protocol_compose():
    model = af.Collection(bulge=linear_gaussian())
    spec = GraphSpec.from_model(model, solved_paths=("bulge.flux",))

    assert [row.name for row in spec.node(("bulge",)).rows] == [
        "centre",
        "sigma",
        "intensity",
        "flux",
    ]
    assert spec.counts["solved"] == 2


# ---------------------------------------------------------------- rule R7


def test_a_float_subclass_model_is_a_row_on_its_owner():
    """
    Rule R7 widened: ``af.Model(Redshift)`` -- a ``float`` subclass -- is the
    ordinary ``redshift`` pill on the galaxy that owns it, never a one-pill
    child card.  The row is named for the attribute; the class name is not shown.
    """
    redshift = af.Model(Redshift)
    # A custom `__new__(cls, ...)` must not make `cls` a model parameter, or the
    # wrapper's own class is overwritten and R7 cannot see the float subclass.
    assert redshift.cls is Redshift
    redshift.redshift = af.UniformPrior(lower_limit=0.0, upper_limit=3.0)

    model = af.Collection(
        galaxies=af.Collection(lens=af.Model(Galaxy, redshift=redshift, bulge=sersic()))
    )
    spec = GraphSpec.from_model(model)
    lens = spec.node(("galaxies", "lens"))

    assert [child.name for child in lens.children] == ["bulge"]
    row = spec.row(("galaxies", "lens", "redshift"))
    assert row.sampling == "free"
    assert row.prior_cls_name == "UniformPrior"
    assert row.in_model_info is True
    assert spec.path_index["galaxies/lens/redshift"] == (
        ("galaxies", "lens", "redshift"),
    )


def test_a_fixed_float_subclass_model_is_a_fixed_row():
    redshift = af.Model(Redshift)
    redshift.redshift = 0.5

    spec = GraphSpec.from_model(af.Model(Galaxy, redshift=redshift, bulge=sersic()))
    row = spec.row(("redshift",))

    assert (row.sampling, row.value) == ("fixed", 0.5)
    assert spec.node(("redshift",)) is None


# ---------------------------------------------------------------- collapse


def test_a_plate_of_solved_members_collapses_and_says_so():
    """
    Thirty linear Gaussians -- the MGE shape.  The solved row is part of every
    member's R1 signature, so the plate still forms, and the ``repeats`` line
    names the state.
    """
    basis = af.Collection(linear_gaussian(0.1 * (index + 1)) for index in range(30))
    spec = GraphSpec.from_model(af.Collection(basis=basis))
    node = spec.node(("basis", "0"))

    assert node.plate is not None
    assert node.plate.count == 30
    assert node.plate.representative_key == "0 - 29"
    (repeats,) = node.plate.repeats
    assert "intensity solved" in repeats
    assert spec.counts["plates"] == 1
    # The count is of the *uncollapsed* tree: thirty solved intensities.
    assert spec.counts["solved"] == 30
    assert spec.path_index["basis/0/intensity"] == ()


# ---------------------------------------------------------------- determinism


def test_the_spec_is_byte_stable_with_solved_rows():
    def _build():
        af.ModelObject._ids = itertools.count()
        af.Prior._ids = itertools.count()
        model = af.Collection(
            bulge=linear_gaussian(),
            point=af.Model(PointSolved),
            gaussian=af.Model(af.ex.Gaussian),
        )
        return json.dumps(
            graph_spec_from(
                model, solved_paths=("gaussian.flux", "nowhere.thing")
            ).to_dict()
        )

    assert _build() == _build()
