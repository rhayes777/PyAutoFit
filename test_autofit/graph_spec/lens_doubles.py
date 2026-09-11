"""
Structural doubles of the lens model classes, for the epic's three acceptance
models.

autofit imports **autonerves only** -- never autogalaxy or autolens, not even in
a test (``AGENTS.md``).  So the acceptance models are built from plain Python
classes defined here that carry the *same class names, attribute names and
constructor signatures* as the real profiles, and are composed exactly as the
current workspace scripts compose them:

* ``autolens_workspace/scripts/imaging/start_here.py`` and
  ``scripts/guides/modeling/cookbook.py`` -- (a) the simple lens;
* ``scripts/imaging/features/multi_gaussian_expansion/modeling.py`` lines
  209-261 -- (b) the MGE;
* ``scripts/group/modeling.py`` lines 288-345 -- (e) the group model.

The doubles have no prior configuration, so ``af.Model(Cls)`` stores a
``ConfigException`` for every parameter (``prior_model.py``
:meth:`Model.make_prior`).  Every parameter is therefore given an explicit prior
here, mirroring the limits in ``pyautogalaxy/autogalaxy/config/priors/*.yaml`` --
with one deliberate exception, ``Hilbert.areas_factor``, which is left unset so
the acceptance can assert the ``missing`` state.

Known divergences from today's libraries, kept because the epic's acceptance
model needs them (see ``PyAutoFit#1605``):

* ``Hilbert`` no longer takes ``areas_factor`` in autoarray -- the modern
  signature is ``(pixels, weight_power, weight_floor)``.  The epic's ``missing``
  case is that parameter, so the double keeps it.
* the group dataset shipped with the workspace has *two* extra-galaxy centres;
  the epic's acceptance model has **eight**, so eight distinct fixed centres are
  used here.
"""

from typing import Optional, Tuple

import autofit as af

# ----------------------------------------------------------------------------
# priors, mirroring pyautogalaxy/autogalaxy/config/priors/*.yaml
# ----------------------------------------------------------------------------


def _gaussian(mean, sigma):
    return af.GaussianPrior(mean=mean, sigma=sigma)


def _uniform(lower, upper):
    return af.UniformPrior(lower_limit=lower, upper_limit=upper)


def _log_uniform(lower, upper):
    return af.LogUniformPrior(lower_limit=lower, upper_limit=upper)


def _ell_comp():
    """``TruncatedGaussian`` (0.0, 0.3) on [-1, 1] -- every ``ell_comps`` slot."""
    return af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0
    )


def _set_tuple(model, name, priors):
    """
    Set a tuple parameter's slots.

    ``Model.__setattr__`` routes an underscored name into its tuple prior by
    splitting on the *first* underscore, so ``model.ell_comps_0 = ...`` looks for
    a tuple prior called ``ell`` and silently creates a stray attribute instead.
    Reaching the ``TuplePrior`` directly is the reliable form.
    """
    tuple_prior = getattr(model, name)
    for index, prior in enumerate(priors):
        setattr(tuple_prior, f"{name}_{index}", prior)


# ----------------------------------------------------------------------------
# the doubles
# ----------------------------------------------------------------------------


class AbstractRegularization:
    """Stand-in for ``aa.AbstractRegularization``, only ever an annotation."""


class Sersic:
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
    ):
        self.centre = centre
        self.ell_comps = ell_comps
        self.intensity = intensity
        self.effective_radius = effective_radius
        self.sersic_index = sersic_index


class SersicSph:
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
    ):
        self.centre = centre
        self.intensity = intensity
        self.effective_radius = effective_radius
        self.sersic_index = sersic_index


class SersicCore:
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
        radius_break: float = 0.025,
        intensity: float = 0.05,
        gamma: float = 0.25,
        alpha: float = 3.0,
    ):
        self.centre = centre
        self.ell_comps = ell_comps
        self.effective_radius = effective_radius
        self.sersic_index = sersic_index
        self.radius_break = radius_break
        self.intensity = intensity
        self.gamma = gamma
        self.alpha = alpha


class Gaussian:
    """``al.lp_linear.Gaussian`` -- linear, so there is no ``intensity``."""

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        sigma: float = 1.0,
    ):
        self.centre = centre
        self.ell_comps = ell_comps
        self.sigma = sigma


class Basis:
    def __init__(
        self,
        profile_list: list = None,
        regularization: Optional[AbstractRegularization] = None,
    ):
        self.profile_list = profile_list
        self.regularization = regularization


class Isothermal:
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
    ):
        self.centre = centre
        self.ell_comps = ell_comps
        self.einstein_radius = einstein_radius


class ExternalShear:
    def __init__(self, gamma_1: float = 0.0, gamma_2: float = 0.0):
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2


class dPIEMassSph:
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        sigma: float = 200.0,
        r_core: float = 0.1,
        r_cut: float = 20.0,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
        H0: float = 67.66,
        Om0: float = 0.30966,
    ):
        self.centre = centre
        self.sigma = sigma
        self.r_core = r_core
        self.r_cut = r_cut
        self.redshift_object = redshift_object
        self.redshift_source = redshift_source
        self.H0 = H0
        self.Om0 = Om0


class Hilbert:
    """
    The epic-era ``al.image_mesh.Hilbert``.  ``areas_factor`` is deliberately
    left without a prior wherever this is used, so it sits in the tree as a
    ``ConfigException`` -- the ``missing`` state.
    """

    def __init__(self, pixels: int = 1000, areas_factor: float = 1.0):
        self.pixels = pixels
        self.areas_factor = areas_factor


class Delaunay:
    def __init__(self):
        pass


class AdaptiveBrightnessSplit:
    def __init__(
        self,
        inner_coefficient: float = 1.0,
        outer_coefficient: float = 1.0,
        signal_scale: float = 0.1,
    ):
        self.inner_coefficient = inner_coefficient
        self.outer_coefficient = outer_coefficient
        self.signal_scale = signal_scale


class Pixelization:
    def __init__(self, image_mesh=None, mesh=None, regularization=None):
        self.image_mesh = image_mesh
        self.mesh = mesh
        self.regularization = regularization


class Galaxy(af.ModelObject):
    """``al.Galaxy`` -- a ``ModelObject`` whose extra kwargs become attributes."""

    def __init__(self, redshift: float, **kwargs):
        super().__init__()
        self.redshift = redshift
        for name, value in kwargs.items():
            setattr(self, name, value)


# ----------------------------------------------------------------------------
# configured models
# ----------------------------------------------------------------------------


def sersic():
    model = af.Model(Sersic)
    _set_tuple(model, "centre", (_gaussian(0.0, 0.3), _gaussian(0.0, 0.3)))
    _set_tuple(model, "ell_comps", (_ell_comp(), _ell_comp()))
    model.intensity = _log_uniform(1e-06, 1e06)
    model.effective_radius = _uniform(0.0, 30.0)
    model.sersic_index = _uniform(0.8, 5.0)
    return model


def sersic_sph():
    model = af.Model(SersicSph)
    _set_tuple(model, "centre", (_gaussian(0.0, 0.3), _gaussian(0.0, 0.3)))
    model.intensity = _log_uniform(1e-06, 1e06)
    model.effective_radius = _uniform(0.0, 30.0)
    model.sersic_index = _uniform(0.8, 5.0)
    return model


def sersic_core():
    model = af.Model(SersicCore)
    _set_tuple(model, "centre", (_gaussian(0.0, 0.3), _gaussian(0.0, 0.3)))
    _set_tuple(model, "ell_comps", (_ell_comp(), _ell_comp()))
    model.effective_radius = _uniform(0.0, 30.0)
    model.sersic_index = _uniform(0.8, 5.0)
    # `radius_break`, `gamma` and `alpha` are `Constant` in the real config.
    model.radius_break = 0.025
    model.intensity = _log_uniform(1e-05, 1000.0)
    model.gamma = 0.25
    model.alpha = 3.0
    return model


def linear_gaussian():
    model = af.Model(Gaussian)
    _set_tuple(model, "centre", (_gaussian(0.0, 0.3), _gaussian(0.0, 0.3)))
    _set_tuple(model, "ell_comps", (_ell_comp(), _ell_comp()))
    model.sigma = _uniform(0.0, 25.0)
    return model


def isothermal():
    model = af.Model(Isothermal)
    _set_tuple(model, "centre", (_gaussian(0.0, 0.1), _gaussian(0.0, 0.1)))
    _set_tuple(model, "ell_comps", (_ell_comp(), _ell_comp()))
    model.einstein_radius = _uniform(0.0, 8.0)
    return model


def external_shear():
    model = af.Model(ExternalShear)
    model.gamma_1 = _uniform(-0.3, 0.3)
    model.gamma_2 = _uniform(-0.3, 0.3)
    return model


def dpie_mass_sph():
    model = af.Model(dPIEMassSph)
    _set_tuple(model, "centre", (_gaussian(0.0, 0.1), _gaussian(0.0, 0.1)))
    model.sigma = _uniform(0.0, 1000.0)
    model.r_core = _uniform(0.0, 10.0)
    model.r_cut = _uniform(0.0, 100.0)
    model.redshift_object = _uniform(0.0, 1.0)
    model.redshift_source = _uniform(0.0, 1.0)
    model.H0 = _uniform(0.0, 100.0)
    model.Om0 = _uniform(0.0, 1.0)
    return model


# ----------------------------------------------------------------------------
# (a) the simple lens
# ----------------------------------------------------------------------------


def simple_lens_model():
    """
    ``lens = Galaxy(redshift=0.5, bulge=Sersic, mass=Isothermal,
    shear=ExternalShear)``, ``source = Galaxy(redshift=1.0, bulge=Sersic)``.
    """
    lens = af.Model(
        Galaxy,
        redshift=0.5,
        bulge=sersic(),
        mass=isothermal(),
        shear=external_shear(),
    )
    source = af.Model(Galaxy, redshift=1.0, bulge=sersic())
    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


# ----------------------------------------------------------------------------
# (b) the MGE, 2 x 30, with a pixelized source
# ----------------------------------------------------------------------------


def mge_model(total_gaussians: int = 30, gaussian_per_basis: int = 2):
    """
    The MGE composition of
    ``scripts/imaging/features/multi_gaussian_expansion/modeling.py``: two bases
    of ``total_gaussians`` linear Gaussians, one shared ``centre`` across *all*
    of them, one shared ``ell_comps`` *per basis*, and a per-Gaussian fixed
    ``sigma``.
    """
    log10_sigma_list = [
        -1.0 + (index * (1.0 - -1.0) / (total_gaussians - 1))
        for index in range(total_gaussians)
    ]

    centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

    bulge_gaussian_list = []
    for _ in range(gaussian_per_basis):
        gaussian_list = af.Collection(linear_gaussian() for _ in range(total_gaussians))
        for index, gaussian in enumerate(gaussian_list):
            gaussian.centre.centre_0 = centre_0
            gaussian.centre.centre_1 = centre_1
            gaussian.ell_comps = gaussian_list[0].ell_comps
            gaussian.sigma = 10 ** log10_sigma_list[index]
        bulge_gaussian_list += gaussian_list

    bulge = af.Model(Basis, profile_list=bulge_gaussian_list)

    lens = af.Model(
        Galaxy,
        redshift=0.5,
        bulge=bulge,
        mass=isothermal(),
        shear=external_shear(),
    )

    image_mesh = af.Model(Hilbert, pixels=1000)
    # `areas_factor` is deliberately left as its `ConfigException`.
    mesh = af.Model(Delaunay)
    regularization = af.Model(AdaptiveBrightnessSplit)
    regularization.inner_coefficient = _log_uniform(1e-04, 1e04)
    regularization.outer_coefficient = _log_uniform(1e-04, 1e04)
    regularization.signal_scale = _uniform(0.0, 10.0)

    source = af.Model(
        Galaxy,
        redshift=1.0,
        pixelization=af.Model(
            Pixelization,
            image_mesh=image_mesh,
            mesh=mesh,
            regularization=regularization,
        ),
    )
    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


# ----------------------------------------------------------------------------
# (e) the group model, eight extra galaxies
# ----------------------------------------------------------------------------

#: Eight distinct fixed ``(y, x)`` centres -- the epic's acceptance model.  The
#: dataset shipped with `scripts/group/modeling.py` has two.
EXTRA_GALAXY_CENTRES = (
    (1.0, 3.5),
    (-2.0, -3.5),
    (3.1, -1.4),
    (-1.7, 2.6),
    (0.6, -4.2),
    (4.3, 0.9),
    (-3.8, -0.4),
    (2.2, 2.2),
)


def group_model(centres=EXTRA_GALAXY_CENTRES):
    """The composition of ``scripts/group/modeling.py`` lines 288-345."""
    lens_0 = af.Model(
        Galaxy,
        redshift=0.5,
        bulge=sersic(),
        mass=isothermal(),
        shear=external_shear(),
    )

    extra_galaxies_list = []
    for centre in centres:
        mass = dpie_mass_sph()
        mass.centre = centre
        mass.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=300.0)
        mass.r_core = 0.0
        mass.r_cut = 10.0
        mass.redshift_object = 0.5
        mass.redshift_source = 1.0
        mass.H0 = 67.66
        mass.Om0 = 0.30966

        extra_galaxies_list.append(
            af.Model(Galaxy, redshift=0.5, bulge=sersic_sph(), mass=mass)
        )

    extra_galaxies = af.Collection(extra_galaxies_list)
    source = af.Model(Galaxy, redshift=1.0, bulge=sersic_core())

    return af.Collection(
        galaxies=af.Collection(lens_0=lens_0, source=source),
        extra_galaxies=extra_galaxies,
    )
