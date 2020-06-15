import inspect
import math
import typing

from autoconf import conf
import autofit as af
# noinspection PyAbstractClass
from autofit.mapper.prior_model import attribute_pair
from autofit.tools.phase import Dataset


class MockResult:
    def __init__(
        self,
        samples=None,
        instance=None,
        model=None,
        analysis=None,
        search=None,
    ):

        self.instance = instance or af.ModelInstance()
        self.model = model or af.ModelMapper()
        self.samples = samples or MockSamples(max_log_likelihood_instance=self.instance)

        self.previous_model = model
        self.gaussian_tuples = None
        self.analysis = analysis
        self.search = search

    def model_absolute(self, absolute):
        return self.model

    def model_relative(self, relative):
        return self.model

    @property
    def last(self):
        return self


class MockSamples(af.PDFSamples):
    def __init__(
        self,
        max_log_likelihood_instance=None,
        log_likelihoods=None,
        gaussian_tuples=None,
    ):

        if log_likelihoods is None:
            log_likelihoods = [1.0, 2.0, 3.0]

        super().__init__(
            model=None,
            parameters=[],
            log_likelihoods=log_likelihoods,
            log_priors=[],
            weights=[],
        )

        self._max_log_likelihood_instance = max_log_likelihood_instance
        self.gaussian_tuples = gaussian_tuples

    @property
    def max_log_likelihood_instance(self):
        return self._max_log_likelihood_instance

    def gaussian_priors_at_sigma(self, sigma=None):
        return self.gaussian_tuples


class MockSearch(af.NonLinearSearch):
    def __init__(self, paths=None, samples=None):

        super().__init__(paths=paths)

        self.samples = samples or MockSamples()

    def _fit(self, model, analysis):
        class Fitness:
            def __init__(self, instance_from_vector):
                self.result = None
                self.instance_from_vector = instance_from_vector

            def __call__(self, vector):
                instance = self.instance_from_vector(vector)

                log_likelihood = analysis.log_likelihood_function(instance)
                self.result = MockResult(instance=instance)

                # Return Chi squared
                return -2 * log_likelihood

        fitness_function = Fitness(model.instance_from_vector)
        fitness_function(model.prior_count * [0.8])

        return fitness_function.result

    @property
    def config_type(self):
        return conf.instance.mock

    @property
    def tag(self):
        return "mock"

    def samples_from_model(self, model):
        return self.samples


class MockDataset(Dataset):
    @property
    def metadata(self) -> dict:
        return dict()

    @property
    def name(self) -> str:
        return "name"


### Mock Classes ###

class SimpleClass:
    def __init__(self, one, two: float):
        self.one = one
        self.two = two


class ComplexClass:
    def __init__(self, simple: SimpleClass):
        self.simple = simple


class ListClass:
    def __init__(self, ls: list):
        self.ls = ls


class MockDistance(af.DimensionType):
    pass


class MockDistanceClass:
    @af.map_types
    def __init__(self, first: MockDistance, second: MockDistance):
        self.first = first
        self.second = second


class PositionClass:
    @af.map_types
    def __init__(self, position: typing.Tuple[MockDistance, MockDistance]):
        self.position = position


class DeferredClass:
    def __init__(self, one, two):
        self.one = one
        self.two = two


class RelativeWidth:
    def __init__(self, one, two, three):
        self.one = one
        self.two = two
        self.three = three


class MockClassx2:
    def __init__(self, one=1, two=2):
        self.one = one
        self.two = two


class MockClassx4:
    def __init__(self, one=1, two=2, three=3, four=4):
        self.one = one
        self.two = two
        self.three = three
        self.four = four


class MockProfile:
    def __init__(self, centre=(0.0, 0.0), intensity=0.1):
        self.centre = centre
        self.intensity = intensity


class MockClassInf:
    def __init__(self, one, two):
        self.one = one
        self.two = two


### Real Classes ###

class Circle:
    def __init__(self, radius):
        self.radius = radius

    def with_circumference(self, circumference):
        self.circumference = circumference

    @property
    def circumference(self):
        return self.radius * 2 * math.pi

    @circumference.setter
    def circumference(self, circumference):
        self.radius = circumference / (2 * math.pi)


class Galaxy:
    def __init__(
            self,
            light_profiles: list = None,
            mass_profiles: list = None,
            redshift=None,
            **kwargs
    ):
        self.redshift = redshift
        self.light_profiles = light_profiles
        self.mass_profiles = mass_profiles
        self.kwargs = kwargs


class HyperGalaxy:
    pass


class Redshift:
    def __init__(self, redshift):
        self.redshift = redshift


# noinspection PyAbstractClass
class GalaxyModel(af.AbstractPriorModel):
    def instance_for_arguments(self, arguments):
        try:
            return Galaxy(redshift=self.redshift.instance_for_arguments(arguments))
        except AttributeError:
            return Galaxy()

    def __init__(self, model_redshift=False, **kwargs):
        super().__init__()
        self.redshift = af.PriorModel(Redshift) if model_redshift else None
        print(self.redshift)
        self.__dict__.update(
            {
                key: af.PriorModel(value) if inspect.isclass(value) else value
                for key, value in kwargs.items()
            }
        )

    @property
    def instance_tuples(self):
        return []

    @property
    @attribute_pair.cast_collection(
        attribute_pair.PriorNameValue
    )
    def unique_prior_tuples(self):
        return (
            [item for item in self.__dict__.items() if isinstance(item[1], af.Prior)]
            + [("redshift", self.redshift.redshift)]
            if self.redshift is not None
            else []
        )

    @property
    @attribute_pair.cast_collection(af.PriorModelNameValue)
    def flat_prior_model_tuples(self):
        return [
            item
            for item in self.__dict__.items()
            if isinstance(item[1], af.AbstractPriorModel)
        ]


class GeometryProfile:
    def __init__(self, centre=(0.0, 0.0)):
        """Abstract GeometryProfile, describing an object with y, x cartesian
        coordinates """
        self.centre = centre

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class SphericalProfile(GeometryProfile):
    def __init__(self, centre=(0.0, 0.0)):
        """ Generic circular profiles class to contain functions shared by light and
        mass profiles.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) coordinates of the origin of the profile.
        """
        super(SphericalProfile, self).__init__(centre)


class EllipticalProfile(SphericalProfile):
    def __init__(self, centre=(0.0, 0.0),             elliptical_comps=(0.0, 0.0)):
        """ Generic elliptical profiles class to contain functions shared by light
        and mass profiles.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) coordinates of the origin of the profiles
        axis_ratio : float
            Ratio of profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis
        """
        super(EllipticalProfile, self).__init__(centre)
        self.axis_ratio = axis_ratio
        self.phi = phi


class MassProfile:
    def surface_density_func(self, eta):
        raise NotImplementedError("surface_density_at_radius should be overridden")

    def surface_density_from_grid(self, grid):
        pass

    def potential_from_grid(self, grid):
        pass

    def deflections_from_grid(self, grid):
        raise NotImplementedError("deflections_from_grid should be overridden")


# noinspection PyAbstractClass
class EllipticalMassProfile(EllipticalProfile, MassProfile):
    def __init__(self, centre=(0.0, 0.0),             elliptical_comps=(0.0, 0.0)):
        """
        Abstract class for elliptical mass profiles.

        Parameters
        ----------
        centre: (float, float)
            The origin of the profile
        axis_ratio : float
            Ellipse's minor-to-major axis ratio (b/a)
        phi : float
            Rotation angle of profile's ellipse counter-clockwise from positive x-axis
        """
        super(EllipticalMassProfile, self).__init__(centre, axis_ratio, phi)
        self.axis_ratio = axis_ratio
        self.phi = phi


# noinspection PyAbstractClass
class EllipticalCoredIsothermal(EllipticalProfile):
    def __init__(
            self,
            centre=(0.0, 0.0),
        e1 = 0.0,
        e2 = 0.0,
            einstein_radius=1.0,
            core_radius=0.05,
    ):
        """
        Represents a cored elliptical isothermal density distribution, which is
        equivalent to the elliptical power-law
        density distribution for the value slope=2.0

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the origin of the profiles
        axis_ratio : float
            Elliptical mass profile's minor-to-major axis ratio (b/a)
        phi : float
            Rotation angle of mass profile's ellipse counter-clockwise from positive
            x-axis
        einstein_radius : float
            Einstein radius of power-law mass profiles
        core_radius : float
            The radius of the inner core
        """

        super(EllipticalCoredIsothermal, self).__init__(
            centre, axis_ratio, phi,
        )
        self.einstein_radius = einstein_radius
        self.core_radius = core_radius


class EllipticalSersic(EllipticalProfile):
    def __init__(
            self,
            centre=(0.0, 0.0),
        e1 = 0.0,
        e2 = 0.0,
            intensity=0.1,
            effective_radius=0.6,
            sersic_index=4.0,
    ):
        """ The elliptical Sersic profile, used for fitting a model_galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) origin of the light profile.
        axis_ratio : float
            Ratio of light profiles ellipse's minor and major axes (b/a).
        phi : float
            Rotation angle of light profile counter-clockwise from positive x-axis.
        intensity : float
            Overall intensity normalisation of the light profiles (electrons per
            second).
        effective_radius : float
            The circular radius containing half the light of this profile.
        sersic_index : Int
            Controls the concentration of the of the light profile.
        """
        super().__init__(
            centre=centre, axis_ratio=axis_ratio, phi=phi,
        )
        self.intensity = intensity
        self.effective_radius = effective_radius
        self.sersic_index = sersic_index


class EllipticalCoreSersic(EllipticalSersic):
    def __init__(
            self,
            centre=(0.0, 0.0),
        e1 = 0.0,
        e2 = 0.0,
            intensity=0.1,
            effective_radius=0.6,
            sersic_index=4.0,
            radius_break=0.01,
            intensity_break=0.05,
            gamma=0.25,
            alpha=3.0,
    ):
        """ The elliptical cored-Sersic profile, used for fitting a model_galaxy's
        light.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) origin of the light profile.
        axis_ratio : float
            Ratio of light profiles ellipse's minor and major axes (b/a).
        phi : float
            Rotation angle of light profile counter-clockwise from positive x-axis.
        intensity : float
            Overall intensity normalisation of the light profiles (electrons per
            second).
        effective_radius : float
            The circular radius containing half the light of this profile.
        sersic_index : Int
            Controls the concetration of the of the light profile.
        radius_break : Float
            The break radius separating the inner power-law (with logarithmic slope
            gamma) and outer Sersic function.
        intensity_break : Float
            The intensity at the break radius.
        gamma : Float
            The logarithmic power-law slope of the inner core profiles
        alpha :
            Controls the sharpness of the transition between the inner core / outer
            Sersic profiles.
        """
        super(EllipticalCoreSersic, self).__init__(
            centre, axis_ratio, phi, intensity, effective_radius, sersic_index
        )
        self.radius_break = radius_break
        self.intensity_break = intensity_break
        self.alpha = alpha
        self.gamma = gamma


class EllipticalExponential(EllipticalSersic):
    def __init__(
            self,
            centre=(0.0, 0.0),
        e1 = 0.0,
        e2 = 0.0,
            intensity=0.1,
            effective_radius=0.6,
    ):
        """ The elliptical exponential profile, used for fitting a model_galaxy's light.

        This is a subset of the elliptical Sersic profile, specific to the case that
        sersic_index = 1.0.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) origin of the light profile.
        axis_ratio : float
            Ratio of light profiles ellipse's minor and major axes (b/a).
        phi : float
            Rotation angle of light profile counter-clockwise from positive x-axis.
        intensity : float
            Overall intensity normalisation of the light profiles (electrons per
            second).
        effective_radius : float
            The circular radius containing half the light of this profile.
        """
        super(EllipticalExponential, self).__init__(
            centre, axis_ratio, phi, intensity, effective_radius, 1.0
        )


class Tracer:
    def __init__(self, lens_galaxy: Galaxy, source_galaxy: Galaxy, grid):
        self.lens_galaxy = lens_galaxy
        self.source_galaxy = source_galaxy
        self.grid = grid