import numpy as np
import pytest

import autofit as af


def _polar(comp_0, comp_1, **kwargs):
    return {
        "strength": np.hypot(comp_0, comp_1),
        "angle": np.degrees(np.arctan2(comp_0, comp_1)),
    }


def _polar_periodic(comp_0, comp_1, **kwargs):
    return {
        "strength": np.hypot(comp_0, comp_1),
        "angle": af.Periodic(np.degrees(np.arctan2(comp_0, comp_1)), period=360.0),
    }


class MockPeriodicComponent:
    latent_parameters = {("comp_0", "comp_1"): _polar_periodic}

    def __init__(self, centre=0.0, comp=(0.0, 0.0)):
        self.centre = centre
        self.comp = comp


def _scaled(comp_0, comp_1, scale, **kwargs):
    return {"scaled": np.hypot(comp_0, comp_1) * scale}


class MockComponent:
    # Sampled as two components, read as a strength and an angle.
    latent_parameters = {("comp_0", "comp_1"): _polar}

    def __init__(self, centre=0.0, comp=(0.0, 0.0)):
        self.centre = centre
        self.comp = comp


class MockScaledComponent(MockComponent):
    # A subclass adds to, and may override, what its base declares.
    latent_parameters = {("comp_0", "comp_1"): _scaled}

    def __init__(self, centre=0.0, comp=(0.0, 0.0), scale=1.0):
        super().__init__(centre=centre, comp=comp)
        self.scale = scale


class MockGalaxy:
    def __init__(self, light=None, mass=None):
        self.light = light
        self.mass = mass


def _component(cls, comp=None, **fixed):
    # The mocks have no prior config, so their priors are given explicitly;
    # assigning the tuple prior after construction keeps ``centre`` first.
    from autofit.mapper.prior.tuple_prior import TuplePrior

    model = af.Model(cls, centre=af.UniformPrior(-1.0, 1.0), **fixed)
    if comp is None:
        comp = TuplePrior()
        comp.comp_0 = af.UniformPrior(-1.0, 1.0)
        comp.comp_1 = af.UniformPrior(-1.0, 1.0)
    model.comp = comp
    return model


def test__latent_parameters__replace_components_in_place():
    model = _component(MockComponent)
    samples = np.array([[0.5, 3.0, 4.0], [0.1, 0.0, 1.0]])

    latent = model.latent_parameters_from(samples)

    assert latent.has_latent
    assert latent.names == ["centre", "strength", "angle"]
    assert latent.latent_mask == [False, True, True]
    assert latent.parameter_array[:, 0] == pytest.approx([0.5, 0.1])
    assert latent.parameter_array[:, 1] == pytest.approx([5.0, 1.0])
    assert latent.parameter_array[:, 2] == pytest.approx([36.869898, 0.0])


def test__latent_parameters__fixed_values_reach_the_conversion_and_subclass_overrides():
    model = _component(MockScaledComponent, scale=2.0)

    latent = model.latent_parameters_from(np.array([[0.0, 3.0, 4.0]]))

    assert latent.names == ["centre", "scaled"]
    assert latent.parameter_array[0, 1] == pytest.approx(10.0)


def test__latent_parameters__no_conversion_when_a_replaced_parameter_is_fixed():
    model = _component(MockComponent, comp=(0.0, 1.0))

    latent = model.latent_parameters_from(np.array([[0.5]]))

    assert not latent.has_latent
    assert not model.has_latent_parameters
    assert latent.names == ["centre"]
    assert latent.parameter_array.tolist() == [[0.5]]


def test__latent_parameters__carry_their_component_superscript_and_grouping():
    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(MockGalaxy, mass=_component(MockComponent)),
            source=af.Model(MockGalaxy, light=af.Model(af.m.MockClassx2NoSuperScript)),
        )
    )
    rng = np.random.default_rng(0)

    latent = model.latent_parameters_from(rng.normal(size=(4, model.prior_count)))

    assert latent.names == ["centre", "strength", "angle", "one", "two"]
    assert latent.superscripts == ["lens.mass"] * 3 + ["source.light"] * 2
    assert latent.parameter_indices_by_component == {
        "lens": [0, 1, 2],
        "source": [3, 4],
    }
    assert latent.parameter_labels_with_superscripts_latex[1] == r"$s^{\rm lens.mass}$"


def test__latent_parameters__plain_model_passes_through():
    model = af.Model(af.m.MockClassx2NoSuperScript)
    samples = np.array([[1.0, 2.0]])

    latent = model.latent_parameters_from(samples)

    assert not latent.has_latent
    assert latent.names == model.parameter_names
    assert latent.parameter_array.tolist() == samples.tolist()


def test__periodic__recentres_on_the_circular_mean():
    from autofit.mapper.prior_model.latent_parameters import Periodic

    # A posterior peaked at the branch cut of [-180, 180): half the samples
    # just below 180, half just above -180. Recentred, they are contiguous
    # around 180 and nothing is left at -180.
    values = np.array([175.0, 178.0, -179.0, -176.0])

    recentred = Periodic(values, period=360.0).recentred()

    assert recentred == pytest.approx([175.0, 178.0, 181.0, 184.0])

    # Weights centre on the posterior: with all the weight on one side the
    # other side is the one that moves.
    recentred = Periodic(values, period=360.0).recentred(weights=[1, 1, 0, 0])
    assert recentred == pytest.approx([175.0, 178.0, 181.0, 184.0])
    recentred = Periodic(values, period=360.0).recentred(weights=[0, 0, 1, 1])
    assert recentred == pytest.approx([-185.0, -182.0, -179.0, -176.0])


def test__latent_parameters__periodic_angle_is_recentred_with_the_sample_weights():
    model = _component(MockPeriodicComponent)
    # arctan2(comp_0, comp_1) for these straddles +-180.
    samples = np.array([[0.0, 0.1, -1.0], [0.0, -0.1, -1.0]])

    latent = model.latent_parameters_from(samples, weights=[1.0, 1.0], recentre=True)

    angle = latent.parameter_array[:, 2]
    assert np.ptp(angle) == pytest.approx(11.42, abs=0.01)  # not ~348
    assert latent.names == ["centre", "strength", "angle"]


def test__latent_parameters__periodic_angle_keeps_the_conversion_range_by_default():
    # The latent output is not recentred, so a column has the range the
    # conversion gives it in every fit, whatever this fit's posterior.
    model = _component(MockPeriodicComponent)
    samples = np.array([[0.0, 0.1, -1.0], [0.0, -0.1, -1.0]])

    latent = model.latent_parameters_from(samples, weights=[1.0, 1.0])

    angle = latent.parameter_array[:, 2]
    assert angle == pytest.approx(np.degrees(np.arctan2([0.1, -0.1], [-1.0, -1.0])))
    assert np.ptp(angle) > 340.0


def test__latent_parameters__grouping_by_latent_component_and_full_paths():
    # The source has no latent parameter, so it has no group of its own in
    # the latent grouping; every parameter keeps its full model path, a latent
    # one under the component that declared it.
    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(MockGalaxy, mass=_component(MockComponent)),
            source=af.Model(MockGalaxy, light=af.Model(af.m.MockClassx2NoSuperScript)),
        )
    )
    rng = np.random.default_rng(1)
    samples = rng.normal(size=(4, model.prior_count))

    latent = model.latent_parameters_from(samples)

    assert latent.parameter_indices_by_latent_component == {"lens": [0, 1, 2]}
    assert latent.paths == [
        ("galaxies", "lens", "mass", "centre"),
        ("galaxies", "lens", "mass", "strength"),
        ("galaxies", "lens", "mass", "angle"),
        ("galaxies", "source", "light", "one"),
        ("galaxies", "source", "light", "two"),
    ]
    items = latent.latent_items
    assert [path for path, _ in items] == latent.paths[1:3]
    assert items[0][1] == pytest.approx(latent.parameter_array[:, 1])


def test__latent_parameters__shared_prior_takes_the_path_of_its_samples_column():
    # The replaced priors are shared by two components; the latent ones are
    # filed under the component the column is named after (the last holder),
    # not under whichever path happens to come first.
    mass = _component(MockComponent)
    multipole = _component(MockComponent, comp=mass.comp)
    multipole.centre = mass.centre
    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(MockGalaxy, light=multipole, mass=mass),
        )
    )

    latent = model.latent_parameters_from(np.array([[0.5, 3.0, 4.0]]))

    assert latent.names == ["centre", "strength", "angle"]
    assert latent.superscripts == ["lens.mass"] * 3
    assert latent.paths == [
        ("galaxies", "lens", "mass", "centre"),
        ("galaxies", "lens", "mass", "strength"),
        ("galaxies", "lens", "mass", "angle"),
    ]


class MockMultipole:
    # Holds the ellipticity of the mass it perturbs (linked to it) and declares
    # only its own components.
    latent_parameters = {
        (
            "multipole_comps_0",
            "multipole_comps_1",
        ): lambda multipole_comps_0, multipole_comps_1, **kwargs: _polar(
            multipole_comps_0, multipole_comps_1
        )
    }

    def __init__(self, ell_comps=(0.0, 0.0), multipole_comps=(0.0, 0.0)):
        self.ell_comps = ell_comps
        self.multipole_comps = multipole_comps


class MockMass:
    latent_parameters = {
        ("ell_comps_0", "ell_comps_1"): lambda ell_comps_0, ell_comps_1, **kwargs: {
            "axis_ratio": 1.0 - np.hypot(ell_comps_0, ell_comps_1),
            "angle": np.degrees(np.arctan2(ell_comps_0, ell_comps_1)),
        }
    }

    def __init__(self, ell_comps=(0.0, 0.0)):
        self.ell_comps = ell_comps


def _tuple_prior(*names):
    from autofit.mapper.prior.tuple_prior import TuplePrior

    prior = TuplePrior()
    for name in names:
        setattr(prior, name, af.UniformPrior(-1.0, 1.0))
    return prior


@pytest.mark.parametrize("mass_first", [True, False])
def test__latent_parameters__shared_prior_is_converted_by_every_holder(mass_first):
    # A power law's ellipticity linked into its multipole: whichever component
    # the priors' samples column is named after, the mass still converts its
    # ellipticity to an axis ratio and angle, and the multipole its own
    # components.
    mass = af.Model(MockMass)
    mass.ell_comps = _tuple_prior("ell_comps_0", "ell_comps_1")
    multipole = af.Model(MockMultipole)
    multipole.ell_comps = mass.ell_comps
    multipole.multipole_comps = _tuple_prior("multipole_comps_0", "multipole_comps_1")
    components = (
        dict(mass=mass, multipole=multipole)
        if mass_first
        else dict(multipole=multipole, mass=mass)
    )
    model = af.Collection(lens=af.Model(MockGalaxyMany, **components))

    latent = model.latent_parameters_from(np.array([[0.3, 0.4, 0.0, 0.5]]))

    assert sorted(latent.names) == ["angle", "angle", "axis_ratio", "strength"]
    by_path = dict(zip(latent.paths, latent.parameter_array[0]))
    assert by_path[("lens", "mass", "axis_ratio")] == pytest.approx(0.5)
    assert by_path[("lens", "mass", "angle")] == pytest.approx(
        np.degrees(np.arctan2(0.3, 0.4))
    )
    assert by_path[("lens", "multipole", "strength")] == pytest.approx(0.5)
    assert (
        dict(zip(latent.paths, latent.superscripts))[("lens", "mass", "axis_ratio")]
        == "lens.mass"
    )
    assert (
        dict(zip(latent.paths, latent.superscripts))[("lens", "multipole", "strength")]
        == "lens.multipole"
    )


class MockGalaxyMany:
    def __init__(self, mass=None, multipole=None):
        self.mass = mass
        self.multipole = multipole


def test__latent_parameters__root_group_does_not_collide_with_a_component_named_model():
    model = af.Collection(
        model=af.Model(af.m.MockClassx2NoSuperScript),
    )
    model.free = af.UniformPrior(0.0, 1.0)

    latent = model.latent_parameters_from(np.zeros((1, model.prior_count)))

    assert model.parameter_indices_by_component == {"model": [0, 1], "model_root": [2]}
    assert latent.parameter_indices_by_component == {
        "model": [0, 1],
        "model_root": [2],
    }
