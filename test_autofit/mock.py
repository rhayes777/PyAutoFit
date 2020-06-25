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

class ListClass:
    def __init__(self, ls: list):
        self.ls = ls


class MockDistance(af.DimensionType):
    pass


class PositionClass:
    @af.map_types
    def __init__(self, position: typing.Tuple[MockDistance, MockDistance]):
        self.position = position


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


class MockClassx2Tuple:
    def __init__(self, one_tuple=(0.0, 0.0)):
        """Abstract GeometryProfile, describing an object with y, x cartesian
        coordinates """
        self.one_tuple = one_tuple

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class MockClassx3TupleFloat:
    def __init__(self, one_tuple=(0.0, 0.0), two=0.1):
        self.one_tuple = one_tuple
        self.two = two


class MockClassRelativeWidth:
    def __init__(self, one, two, three):
        self.one = one
        self.two = two
        self.three = three


class MockClassInf:
    def __init__(self, one, two):
        self.one = one
        self.two = two


class ComplexClass:
    def __init__(self, simple: MockClassx2):
        self.simple = simple


class MockDistanceClass:
    @af.map_types
    def __init__(self, one: MockDistance, two: MockDistance):
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


class MockComponents:
    def __init__(
            self,
            components_0: list = None,
            components_1: list = None,
            parameter=None,
            **kwargs
    ):
        self.parameter = parameter
        self.group_0 = components_0
        self.group_1 = components_1
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
            return MockComponents(parameter=self.redshift.instance_for_arguments(arguments))
        except AttributeError:
            return MockComponents()

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


class Tracer:
    def __init__(self, lens_galaxy: MockComponents, source_galaxy: MockComponents, grid):
        self.lens_galaxy = lens_galaxy
        self.source_galaxy = source_galaxy
        self.grid = grid