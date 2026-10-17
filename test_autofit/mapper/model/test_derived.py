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
    derived_parameters = {("comp_0", "comp_1"): _polar_periodic}

    def __init__(self, centre=0.0, comp=(0.0, 0.0)):
        self.centre = centre
        self.comp = comp


def _scaled(comp_0, comp_1, scale, **kwargs):
    return {"scaled": np.hypot(comp_0, comp_1) * scale}


class MockComponent:
    # Sampled as two components, read as a strength and an angle.
    derived_parameters = {("comp_0", "comp_1"): _polar}

    def __init__(self, centre=0.0, comp=(0.0, 0.0)):
        self.centre = centre
        self.comp = comp


class MockScaledComponent(MockComponent):
    # A subclass adds to, and may override, what its base declares.
    derived_parameters = {("comp_0", "comp_1"): _scaled}

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


def test__derived_parameters__replace_components_in_place():
    model = _component(MockComponent)
    samples = np.array([[0.5, 3.0, 4.0], [0.1, 0.0, 1.0]])

    derived = model.derived_parameters_from(samples)

    assert derived.has_derived
    assert derived.names == ["centre", "strength", "angle"]
    assert derived.derived_mask == [False, True, True]
    assert derived.parameter_array[:, 0] == pytest.approx([0.5, 0.1])
    assert derived.parameter_array[:, 1] == pytest.approx([5.0, 1.0])
    assert derived.parameter_array[:, 2] == pytest.approx([36.869898, 0.0])


def test__derived_parameters__fixed_values_reach_the_conversion_and_subclass_overrides():
    model = _component(MockScaledComponent, scale=2.0)

    derived = model.derived_parameters_from(np.array([[0.0, 3.0, 4.0]]))

    assert derived.names == ["centre", "scaled"]
    assert derived.parameter_array[0, 1] == pytest.approx(10.0)


def test__derived_parameters__no_conversion_when_a_replaced_parameter_is_fixed():
    model = _component(MockComponent, comp=(0.0, 1.0))

    derived = model.derived_parameters_from(np.array([[0.5]]))

    assert not derived.has_derived
    assert not model.has_derived_parameters
    assert derived.names == ["centre"]
    assert derived.parameter_array.tolist() == [[0.5]]


def test__derived_parameters__carry_their_component_superscript_and_grouping():
    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(MockGalaxy, mass=_component(MockComponent)),
            source=af.Model(MockGalaxy, light=af.Model(af.m.MockClassx2NoSuperScript)),
        )
    )
    rng = np.random.default_rng(0)

    derived = model.derived_parameters_from(rng.normal(size=(4, model.prior_count)))

    assert derived.names == ["centre", "strength", "angle", "one", "two"]
    assert derived.superscripts == ["lens.mass"] * 3 + ["source.light"] * 2
    assert derived.parameter_indices_by_component == {
        "lens": [0, 1, 2],
        "source": [3, 4],
    }
    assert derived.parameter_labels_with_superscripts_latex[1] == r"$s^{\rm lens.mass}$"


def test__derived_parameters__plain_model_passes_through():
    model = af.Model(af.m.MockClassx2NoSuperScript)
    samples = np.array([[1.0, 2.0]])

    derived = model.derived_parameters_from(samples)

    assert not derived.has_derived
    assert derived.names == model.parameter_names
    assert derived.parameter_array.tolist() == samples.tolist()


def test__periodic__recentres_on_the_circular_mean():
    from autofit.mapper.prior_model.derived import Periodic

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


def test__derived_parameters__periodic_angle_is_recentred_with_the_sample_weights():
    model = _component(MockPeriodicComponent)
    # arctan2(comp_0, comp_1) for these straddles +-180.
    samples = np.array([[0.0, 0.1, -1.0], [0.0, -0.1, -1.0]])

    derived = model.derived_parameters_from(samples, weights=[1.0, 1.0])

    angle = derived.parameter_array[:, 2]
    assert np.ptp(angle) == pytest.approx(11.42, abs=0.01)  # not ~348
    assert derived.names == ["centre", "strength", "angle"]
