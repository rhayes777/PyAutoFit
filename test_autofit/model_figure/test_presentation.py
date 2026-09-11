"""
Layer 2 -- one test per rule of the visual vocabulary.

The vocabulary is the epic's v2 as amended by the independent review
(``PyAutoMind/draft/feature/autofit/model_figures_epic.md``, sections 2 and
"Independent review").  Every rule the review made binding has a test here, and
every assertion is on the :class:`Presentation` -- never on a picture.
"""

import json

import matplotlib

matplotlib.use("Agg")

import autofit as af
from autofit.graph_spec import GraphSpec
from autofit.model_figure.presentation import (
    build_presentation,
    compact_prior,
    prior_summaries,
)

from test_autofit.graph_spec.lens_doubles import (
    group_model,
    mge_model,
    sersic,
    simple_lens_model,
)


def _presentation(model, **kwargs):
    priors = prior_summaries(model) if kwargs.get("detail") == "priors" else None
    collapse = kwargs.pop("collapse", True)
    solved_paths = kwargs.pop("solved_paths", ())
    spec = GraphSpec.from_model(model, collapse=collapse, solved_paths=solved_paths)
    return build_presentation(spec, priors=priors, **kwargs)


def _cards(presentation):
    return list(presentation.walk())


def _card(presentation, key):
    for card in presentation.walk():
        if card.key == key:
            return card
    raise AssertionError(f"no card at {key!r}")


def _pill(presentation, key):
    for pill in presentation.pills():
        if pill.key == key:
            return pill
    raise AssertionError(f"no pill at {key!r}")


def _composite():
    """Sharing, a relation and an assertion in one model."""
    first = af.Model(af.ex.Gaussian)
    second = af.Model(af.ex.Gaussian)
    second.centre = first.centre
    second.sigma = first.sigma * 2.0
    model = af.Collection(a=first, b=second)
    model.add_assertion(first.sigma > 5.0)
    return model


# ---------------------------------------------------------------- structure


def test_root_collection_frame_is_dropped_when_it_adds_no_branching():
    """The review's "remove redundant outer framing where it adds no branching"."""
    presentation = _presentation(af.Collection(a=af.Model(af.ex.Gaussian)))

    assert [card.key for card in presentation.cards] == ["a"]
    assert presentation.cards[0].title == "a · Gaussian"


def test_root_model_keeps_its_card_because_it_owns_rows():
    presentation = _presentation(af.Model(af.ex.Gaussian))

    assert [card.key for card in presentation.cards] == [""]
    assert presentation.cards[0].title == "model · Gaussian"
    assert [pill.text for pill in presentation.cards[0].pills] == [
        "centre",
        "normalization",
        "sigma",
    ]


def test_collection_is_a_frame_titled_with_its_attribute_name():
    presentation = _presentation(simple_lens_model())

    (galaxies,) = presentation.cards
    assert galaxies.kind == "collection"
    assert galaxies.title == "galaxies"
    assert [child.title for child in galaxies.children] == [
        "lens · Galaxy",
        "source · Galaxy",
    ]


def test_declaration_order_is_never_reordered():
    """``model.info`` says ``bulge, mass, shear``; so does the figure."""
    presentation = _presentation(simple_lens_model())
    lens = _card(presentation, "galaxies/lens")

    assert [child.title for child in lens.children] == [
        "bulge · Sersic",
        "mass · Isothermal",
        "shear · ExternalShear",
    ]


def test_every_key_is_the_path_index_key():
    model = simple_lens_model()
    spec = GraphSpec.from_model(model)
    presentation = build_presentation(spec)

    for card in presentation.walk():
        assert card.key in spec.path_index
    for pill in presentation.pills():
        assert pill.key in spec.path_index


def test_to_dict_is_json_stable():
    model = simple_lens_model()
    first = json.dumps(_presentation(model).to_dict())
    second = json.dumps(_presentation(model).to_dict())

    assert first == second


# ---------------------------------------------------------------- plates


def test_plate_badge_counts_components_and_never_reads_times_n():
    """Review point 2: ``x30`` on a plate and on a parameter meant two things."""
    presentation = _presentation(mge_model())
    plate = _card(presentation, "galaxies/lens/bulge/profile_list/0")

    assert plate.kind == "plate"
    assert plate.badge == "30 components"
    assert "×" not in plate.badge
    assert plate.subtitle == "0 - 29"
    assert plate.note.startswith("Gaussian with priors")


def test_plate_note_says_what_is_repeated():
    presentation = _presentation(group_model())
    plate = _card(presentation, "extra_galaxies/0")

    assert plate.badge == "8 components"
    assert plate.subtitle.startswith("0 - 7")
    assert plate.note is not None


# ---------------------------------------------------------------- states


def test_free_pill_is_name_only():
    presentation = _presentation(af.Model(af.ex.Gaussian))
    pill = _pill(presentation, "sigma")

    assert pill.state == "free"
    assert pill.text == "sigma"
    assert pill.badge is None


def test_fixed_pill_is_shown_by_default():
    """Review point 4: fixed values explain model structure."""
    model = af.Model(af.ex.Gaussian)
    model.sigma = 3.0
    presentation = _presentation(model)

    assert _pill(presentation, "sigma").state == "fixed"


def test_fixed_that_varies_by_member_says_so():
    """A single grey pill otherwise suggests one common fixed value."""
    presentation = _presentation(group_model())
    pill = _pill(presentation, "extra_galaxies/0/mass/centre")

    assert pill.state == "fixed-varies"
    assert pill.text == "centre · fixed, varies by member"


def test_missing_configuration_is_its_own_state():
    """``areas_factor`` is unset: absence from the picture is not acceptable."""
    presentation = _presentation(mge_model())
    pill = _pill(presentation, "galaxies/source/pixelization/image_mesh/areas_factor")

    assert pill.state == "missing"
    assert pill.text == "areas_factor · missing"


def test_solved_is_a_state_and_the_legend_qualifies_it():
    presentation = _presentation(
        af.Collection(gaussian=af.Model(af.ex.Gaussian)),
        solved_paths=("gaussian.normalization",),
    )
    pill = _pill(presentation, "gaussian/normalization")

    assert pill.state == "solved"
    assert "solved during fitting" in presentation.legend


def test_relation_pill_carries_its_defining_expression():
    """The expression IS the pill (review: it is the one permitted number)."""
    presentation = _presentation(_composite())
    pill = _pill(presentation, "b/sigma")

    assert pill.state == "relation"
    assert pill.text == "sigma = a.sigma * 2.0"


# ---------------------------------------------------------------- sharing


def test_sharing_is_a_property_not_a_state():
    presentation = _presentation(_composite())

    owner = _pill(presentation, "a/centre")
    reference = _pill(presentation, "b/centre")

    # Both are still *sampled*.
    assert owner.state == "free"
    assert reference.state == "free"
    assert owner.badge == "shared ×2"
    assert reference.badge == "↗ a.centre"


def test_a_later_occurrence_gets_a_link_back_to_its_owner():
    presentation = _presentation(_composite())

    assert any(
        link.source_key == "b/centre"
        and link.target_key == "a/centre"
        and link.kind == "shared"
        for link in presentation.links
    )


def test_group_sharing_inside_a_plate_reads_shared_across_group():
    """Never ``×30``, which means a different thing on a plate."""
    presentation = _presentation(mge_model())

    owner = _pill(presentation, "galaxies/lens/bulge/profile_list/0/centre")
    assert owner.badge == "shared across group"

    # The second basis' `ell_comps` is shared only within its own 30.
    assert (
        _pill(presentation, "galaxies/lens/bulge/profile_list/30/ell_comps").badge
        == "shared across group"
    )
    # ... while its `centre` is the same prior as the first basis' and points at it.
    reference = _pill(presentation, "galaxies/lens/bulge/profile_list/30/centre")
    assert reference.badge.startswith("↗")
    assert reference.badge.endswith("0.centre")


def test_independent_repeated_priors_are_marked_as_such():
    """Distinct prior objects, identical configuration -- repetition, not sharing."""
    presentation = _presentation(group_model())
    pill = _pill(presentation, "extra_galaxies/0/mass/sigma")

    assert pill.state == "free"
    assert pill.badge == "independent"


# ---------------------------------------------------------------- tuples


def test_tuple_pill_carries_a_two_dimensional_cue():
    presentation = _presentation(simple_lens_model())
    pill = _pill(presentation, "galaxies/lens/bulge/centre")

    assert pill.dim2d is True
    assert pill.text == "centre"


def test_a_mixed_tuple_expands_rather_than_flattening():
    model = sersic()
    model.centre.centre_1 = 0.5
    presentation = _presentation(model)

    assert _pill(presentation, "centre/centre_0").state == "free"
    assert _pill(presentation, "centre/centre_1").state == "fixed"
    assert all(pill.key != "centre" for pill in presentation.pills())


# ---------------------------------------------------------------- redshift


def test_fixed_redshift_becomes_a_card_subtitle():
    presentation = _presentation(simple_lens_model())
    lens = _card(presentation, "galaxies/lens")

    assert lens.subtitle == "redshift = 0.5"
    # A float stays a float: `redshift = 1` reads like an index.
    assert _card(presentation, "galaxies/source").subtitle == "redshift = 1.0"
    assert all(pill.key != "galaxies/lens/redshift" for pill in presentation.pills())


def test_free_redshift_keeps_the_ordinary_pill():
    from test_autofit.graph_spec.lens_doubles import Galaxy, sersic

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                Galaxy,
                redshift=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
                bulge=sersic(),
            )
        )
    )
    presentation = _presentation(model)

    assert _card(presentation, "galaxies/lens").subtitle is None
    assert _pill(presentation, "galaxies/lens/redshift").state == "free"


# ---------------------------------------------------------------- detail


def test_detail_priors_adds_the_compact_prior_summary():
    model = af.Model(af.ex.Gaussian)
    presentation = _presentation(model, detail="priors")

    # The numbers are the model's own, never re-derived by the figure.
    assert (
        _pill(presentation, "centre").text == f"centre  {compact_prior(model.centre)}"
    )
    assert _pill(presentation, "sigma").text == f"sigma  {compact_prior(model.sigma)}"
    assert _pill(presentation, "centre").text.startswith("centre  U(")


def test_detail_priors_prints_the_fixed_value():
    model = af.Model(af.ex.Gaussian)
    model.sigma = 3.0
    presentation = _presentation(model, detail="priors")

    assert _pill(presentation, "sigma").text == "sigma = 3"


def test_compact_prior_maps_every_family():
    assert (
        compact_prior(af.UniformPrior(lower_limit=0.0, upper_limit=100.0))
        == "U(0, 100)"
    )
    assert compact_prior(af.GaussianPrior(mean=0.5, sigma=0.1)) == "N(0.5, 0.1)"
    assert (
        compact_prior(af.LogUniformPrior(lower_limit=1e-06, upper_limit=1e06))
        == "LogU(1e-06, 1e+06)"
    )
    assert (
        compact_prior(
            af.TruncatedGaussianPrior(
                mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0
            )
        )
        == "TN(0, 0.3, -1, 1)"
    )


def test_show_fixed_false_hides_pills_and_prints_the_hidden_count():
    presentation = _presentation(mge_model(), show_fixed=False)

    assert presentation.hidden_fixed > 0
    assert f"{presentation.hidden_fixed} fixed parameters hidden" in presentation.footer
    assert all(
        pill.state not in ("fixed", "fixed-varies") for pill in presentation.pills()
    )


def test_max_depth_folds_a_subtree_into_one_row():
    presentation = _presentation(simple_lens_model(), max_depth=2)
    lens = _card(presentation, "galaxies/lens")

    assert lens.children == ()
    folded = [pill for pill in lens.pills if pill.state == "folded"]
    assert [pill.key for pill in folded] == [
        "galaxies/lens/bulge",
        "galaxies/lens/mass",
        "galaxies/lens/shear",
    ]
    assert folded[0].text == "… 1 components / 7 priors"


# ---------------------------------------------------------------- assertions


def test_an_assertion_is_a_compact_constraint_label_not_an_edge():
    presentation = _presentation(_composite())

    (constraint,) = presentation.constraints
    # Labelled `assert`, so it can never be read as a relation, and written the
    # way it was declared -- `add_assertion(first.sigma > 5.0)`.
    assert constraint.text == "assert a.sigma > 5.0"
    assert constraint.left == "a.sigma"
    assert constraint.right == "5.0"
    assert constraint.key == "a"
    assert all(link.kind != "assertion" for link in presentation.links)
    assert "dashed orange = constraint (assertion)" in presentation.legend


def test_the_constraint_legend_entry_appears_only_when_a_constraint_does():
    assert "constraint" not in _presentation(af.Model(af.ex.Gaussian)).legend


def test_an_assertion_between_two_parameters_keeps_the_spec_order():
    """Both sides resolve, so there is nothing to disambiguate -- do not guess."""
    model = af.Model(af.ex.Gaussian)
    model.add_assertion(model.sigma > model.normalization)
    presentation = _presentation(model)

    (constraint,) = presentation.constraints
    assert constraint.text == "assert normalization < sigma"


# ---------------------------------------------------------------- legend, footer


def test_the_footer_is_singular_for_a_count_of_one():
    model = af.Collection(a=af.Model(af.ex.Gaussian), b=af.Model(af.ex.Gaussian))
    model.b.centre = model.a.centre
    presentation = _presentation(model)

    assert "1 shared prior (unique variables, not references)" in presentation.footer
    assert "1 plate standing for 2 components" in presentation.footer
    assert "0 fixed leaf slots" in presentation.footer


def test_the_legend_is_one_line_of_only_the_states_present():
    presentation = _presentation(af.Model(af.ex.Gaussian))

    assert presentation.legend.count("\n") == 0
    assert presentation.legend == "Legend:  free prior"


def test_the_footer_defines_its_counts():
    presentation = _presentation(mge_model())

    assert "16 unique sampled scalars" in presentation.footer
    assert "63 fixed leaf slots" in presentation.footer
    assert "6 shared priors (unique variables, not references)" in presentation.footer
    assert "2 plates standing for 60 components" in presentation.footer
    assert "totals include every hidden and collapsed element" in presentation.footer
