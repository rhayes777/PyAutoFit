"""
The epic's three acceptance models (``PyAutoMind/active/model_figures_1_graph_spec.md``,
"Acceptance -- the three lens models"), built from the structural doubles in
:mod:`lens_doubles` because autofit may never import autogalaxy or autolens.

**Honesty rule.**  The epic's numbers were measured on the human's real science
models in a throwaway prototype.  Where a double faithful to today's workspace
script cannot reproduce one, the *measured* value is pinned with a
``# epic target: N -- measured M ...`` comment and the gap is reported rather
than the double or the rule being bent.  Every *structural* invariant -- plate
counts and splits, shared multiplicities, the ``missing`` state, fixed tuple
rows, ``varies_by_member``, determinism -- is asserted unconditionally.
"""

import json

from autofit.graph_spec import GraphSpec

from . import lens_doubles as doubles


def _boxes(spec):
    """Component *boxes*: the epic counts ``Model`` nodes, not collection frames."""
    return [node for node in spec.components() if node.kind == "model"]


def _plates(spec):
    return [node for node in spec.components() if node.plate is not None]


def _twice(build):
    """The same model extracted twice, as two serialised specs."""
    model = build()
    return (
        json.dumps(GraphSpec.from_model(model).to_dict()),
        json.dumps(GraphSpec.from_model(model).to_dict()),
    )


# ------------------------------------------------------------------ (a)


def test_a_simple_lens():
    model = doubles.simple_lens_model()
    spec = GraphSpec.from_model(model)

    # 8 raw component nodes -- the epic's figure, exactly.
    assert spec.counts["components_raw"] == 8
    # epic target: 48 info lines -- measured 47 on the structural double; see
    # PyAutoFit#1605.
    assert len(model.info.splitlines()) == 47

    # 6 component boxes after collapse: the two `Collection` frames are not
    # boxes, and nothing in this model repeats.
    assert len(_boxes(spec)) == 6
    assert spec.counts["plates"] == 0
    assert spec.counts["components"] == spec.counts["components_raw"] == 8

    assert [(node.path[-1], node.cls_name) for node in _boxes(spec)] == [
        ("lens", "Galaxy"),
        ("bulge", "Sersic"),
        ("mass", "Isothermal"),
        ("shear", "ExternalShear"),
        ("source", "Galaxy"),
        ("bulge", "Sersic"),
    ]
    # Declaration order, `bulge, mass, shear` -- never `shear` above `mass`.
    assert [node.name for node in spec.node(("galaxies", "lens")).children] == [
        "bulge",
        "mass",
        "shear",
    ]

    # epic target: 13 rows -- measured 17 on the structural double (the two
    # `redshift` rows and both `Sersic` intensities are rows here); see
    # PyAutoFit#1605.
    assert len(spec.rows()) == 17
    assert spec.counts["unique_sampled_scalars"] == 21
    assert spec.counts["fixed_leaf_slots"] == 2
    assert spec.counts["shared_priors"] == 0
    assert spec.counts["missing"] == 0


def test_a_simple_lens_is_deterministic():
    first, second = _twice(doubles.simple_lens_model)
    assert first == second


# ------------------------------------------------------------------ (b)


def test_b_mge_two_by_thirty_with_a_pixelized_source():
    model = doubles.mge_model()
    spec = GraphSpec.from_model(model)

    # epic target: 73 raw nodes -- measured 72 on the structural double, which
    # is 73 minus the `af.Model(int)` node R7 drops: today's `Delaunay` takes no
    # `pixels` argument, so the composition no longer has one.  See
    # PyAutoFit#1605.
    assert spec.counts["components_raw"] == 72
    # epic target: 178 info lines -- measured 173 on the structural double.
    assert len(model.info.splitlines()) == 173

    # *** The epic's headline: 11 components after collapse. ***
    assert len(_boxes(spec)) == 11

    # Two x30 plates, split on `ell_comps` -- never one merged x60.
    plates = _plates(spec)
    assert [plate.plate.count for plate in plates] == [30, 30]
    assert [plate.plate.representative_key for plate in plates] == [
        "0 - 29",
        "30 - 59",
    ]
    assert [plate.path for plate in plates] == [
        ("galaxies", "lens", "bulge", "profile_list", "0"),
        ("galaxies", "lens", "bulge", "profile_list", "30"),
    ]
    assert all(plate.cls_name == "Gaussian" for plate in plates)

    # The centre is in every one of the 60, so it does not discriminate; each
    # basis's `ell_comps` is in exactly its own 30, so it does.
    for plate in plates:
        assert len(plate.plate.shared_in_all) == 4
        # The per-Gaussian fixed sigma is named, never silently merged away.
        assert plate.plate.varies_by_member == ("sigma",)
        assert "sigma fixed (varies by member)" in plate.plate.repeats[0]
        assert "centre ⇄ shared" in plate.plate.repeats[0]
        assert "ell_comps ⇄ shared" in plate.plate.repeats[0]

    # 6 shared priors: centre_0 and centre_1 across all 60, then each basis's
    # ell_comps_0 / ell_comps_1 across its own 30.
    assert spec.counts["shared_priors"] == 6
    assert sorted(
        (len(edge.occurrences) for edge in spec.shared), reverse=True
    ) == [60, 60, 30, 30, 30, 30]

    # 16 unique sampled scalars.
    assert spec.counts["unique_sampled_scalars"] == 16

    # `areas_factor` is unset: `missing`, not `solved`, and never absent.
    areas_factor = spec.row(
        ("galaxies", "source", "pixelization", "image_mesh", "areas_factor")
    )
    assert areas_factor is not None
    assert areas_factor.sampling == "missing"
    assert areas_factor.prior_cls_name == "ConfigException"
    assert areas_factor.in_model_info is True
    assert spec.counts["missing"] == 1
    # ... while `pixels` beside it is a plain fixed value.
    pixels = spec.row(("galaxies", "source", "pixelization", "image_mesh", "pixels"))
    assert pixels.sampling == "fixed"
    assert pixels.value == 1000.0

    # Nothing fixed is dropped by the plates: 60 sigmas plus `pixels`, the two
    # redshifts and the `Basis.regularization` slot.
    assert spec.counts["fixed_leaf_slots"] == 64


def test_b_mge_path_index_records_the_finer_partition():
    spec = GraphSpec.from_model(doubles.mge_model())

    # `model.info` groups the shared centre across all 60; the figure's
    # partition is finer (two plates of 30), so the entry records both.
    entry = spec.path_index[
        "galaxies/lens/bulge/profile_list/0/centre/centre_0"
    ]
    assert entry == {
        "figure": ["galaxies/lens/bulge/profile_list/0 - 29/centre/centre_0"],
        "info": ["galaxies/lens/bulge/profile_list/0 - 59/centre/centre_0"],
    }
    # `ell_comps` is shared by exactly one plate's members, so the two agree and
    # the plain tuple-of-paths shape is kept.
    assert spec.path_index[
        "galaxies/lens/bulge/profile_list/30/ell_comps/ell_comps_0"
    ] == (
        (
            "galaxies",
            "lens",
            "bulge",
            "profile_list",
            "30 - 59",
            "ell_comps",
            "ell_comps_0",
        ),
    )


def test_b_mge_is_deterministic():
    first, second = _twice(doubles.mge_model)
    assert first == second


# ------------------------------------------------------------------ (e)


def test_e_group_scale_with_eight_extra_galaxies():
    model = doubles.group_model()
    spec = GraphSpec.from_model(model)

    # epic target: 166 raw nodes -- measured 33 on the structural double.  The
    # epic's group model is bigger than today's `scripts/group/modeling.py`
    # composition (one main lens, one source, eight extra galaxies); see
    # PyAutoFit#1605.
    assert spec.counts["components_raw"] == 33
    # epic target: 418 info lines -- measured 199 on the structural double.
    assert len(model.info.splitlines()) == 199
    # epic target: 15 components (16 before R7) -- measured 9 boxes.
    assert len(_boxes(spec)) == 9

    # *** The structural invariant: the eight extra galaxies are ONE plate. ***
    (plate,) = _plates(spec)
    assert plate.path == ("extra_galaxies", "0")
    assert plate.cls_name == "Galaxy"
    assert plate.plate.count == 8
    assert plate.plate.representative_key == "0 - 7"
    assert plate.plate.member_paths == tuple(
        ("extra_galaxies", str(index)) for index in range(8)
    )

    # The per-galaxy fixed `centre` tuple is a row, and its variation is named.
    assert plate.plate.varies_by_member == ("mass.centre",)
    centre = spec.row(("extra_galaxies", "0", "mass", "centre"))
    assert centre is not None
    assert centre.dimensionality == "tuple"
    assert centre.sampling == "fixed"
    assert [component.name for component in centre.components] == [
        "centre_0",
        "centre_1",
    ]
    assert [component.value for component in centre.components] == [1.0, 3.5]
    # ... and the eight distinct centres are recorded, not hidden behind the
    # representative's.
    assert spec.path_index["extra_galaxies/0/mass/centre"] == {
        "figure": ["extra_galaxies/0 - 7/mass/centre"],
        "info": [
            f"extra_galaxies/{index}/mass/centre" for index in range(8)
        ],
    }

    # Nothing fixed is dropped: every member's slots are still counted.
    # epic target: 291 fixed leaf slots -- measured 77 on the structural double.
    assert spec.counts["fixed_leaf_slots"] == 77
    # epic target: 41 unique sampled scalars -- measured 69.
    assert spec.counts["unique_sampled_scalars"] == 69
    # epic target: 22 shared priors -- measured 0: today's group script shares
    # no prior between galaxies.
    assert spec.counts["shared_priors"] == 0
    assert spec.counts["missing"] == 0


def test_e_group_scale_is_deterministic():
    first, second = _twice(doubles.group_model)
    assert first == second
