import numpy as np

import autofit as af
from autoconf import conf
from autofit.non_linear.samples import Sample
from autofit.tools.phase import Dataset


class MockAnalysis(af.Analysis):
    prior_count = 2

    def __init__(self):
        super().__init__()
        self.fit_instances = list()

    def log_likelihood_function(self, instance):
        self.fit_instances.append(instance)
        return [1]

    def visualize(self, paths, instance, during_analysis):
        pass

    def log(self, instance):
        pass


class MockResult:
    def __init__(
            self,
            samples=None,
            instance=None,
            previous_model=None,
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
        self.previous_model = previous_model

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
            samples=[
                Sample(
                    log_likelihood=log_likelihood,
                    log_prior=0.0,
                    weights=0.0
                )
                for log_likelihood
                in log_likelihoods
            ]
        )

        self._max_log_likelihood_instance = max_log_likelihood_instance
        self.gaussian_tuples = gaussian_tuples

    @property
    def max_log_likelihood_instance(self):
        return self._max_log_likelihood_instance

    def gaussian_priors_at_sigma(self, sigma=None):
        return self.gaussian_tuples

    def write_table(self, filename: str):
        pass

    def info_to_json(self, filename):
        pass


class MockSearch(af.NonLinearSearch):
    def __init__(self, paths=None, samples=None, name=None):
        self.name = name
        super().__init__(paths=paths)

        self.samples = samples or MockSamples()

    def _fit(self, model, analysis, log_likelihood_cap=None):
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

        analysis.save_attributes_for_aggregator(paths=self.paths)

        fitness_function = Fitness(model.instance_from_vector)
        fitness_function(model.prior_count * [0.5])

        return fitness_function.result

    @property
    def config_type(self):
        return conf.instance["non_linear"]["mock"]

    @property
    def tag(self):
        return "mock"

    def perform_update(self, model, analysis, during_analysis):
        self.save_samples(samples=self.samples)
        return self.samples

    def samples_via_csv_json_from_model(self, model):
        return self.samples

    def samples_via_sampler_from_model(self, model):
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


class MockClassx3(MockClassx4):
    def __init__(self, one=1, two=2, three=3):
        super().__init__(one, two, three)


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


class DeferredClass:
    def __init__(self, one, two):
        self.one = one
        self.two = two


class WithFloat:
    def __init__(self, value):
        self.value = value


class WithTuple:
    def __init__(self, tup=(0.0, 0.0)):
        self.tup = tup


### Real Classes ###

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


class MedianPDFInstance:
    def __init__(self, name):
        self.name = name


class MockPhaseOutput:
    def __init__(self, directory, pipeline, phase, dataset):
        self.directory = directory
        self.pipeline = pipeline
        self.phase = phase
        self.dataset = dataset

    @property
    def median_pdf_instance(self):
        return MedianPDFInstance(
            self.phase
        )

    @property
    def output(self):
        return self


class Profile:
    def __init__(self, centre=0.0, intensity=0.01):
        """Represents an Abstract 1D profile.

        Parameters
        ----------
        centre : float
            The x coordinate of the profile centre.
        intensity : float
            Overall intensity normalisation of the profile.
        """
        self.centre = centre
        self.intensity = intensity


class Gaussian(Profile):
    def __init__(
            self,
            centre=0.0,  # <- PyAutoFit recognises these constructor arguments
            intensity=0.1,  # <- are the Gaussian's model parameters.
            sigma=0.01,
    ):
        """Represents a 1D Gaussian profile, which may be treated as a model-component of PyAutoFit the
        parameters of which are fitted for by a non-linear search.

        Parameters
        ----------
        centre : float
            The x coordinate of the profile centre.
        intensity : float
            Overall intensity normalisation of the Gaussian profile.
        sigma : float
            The sigma value controlling the size of the Gaussian.
        """
        super().__init__(centre=centre, intensity=intensity)
        self.sigma = sigma  # We still need to set sigma for the Gaussian, of course.

    def __call__(self, xvalues):
        """
        Calculate the intensity of the profile on a line of Cartesian x coordinates.

        The input xvalues are translated to a coordinate system centred on the Gaussian, using its centre.

        Parameters
        ----------
        xvalues : np.ndarray
            The x coordinates in the original reference frame of the grid.
        """
        transformed_xvalues = np.subtract(xvalues, self.centre)
        return np.multiply(
            np.divide(self.intensity, self.sigma * np.sqrt(2.0 * np.pi)),
            np.exp(-0.5 * np.square(np.divide(transformed_xvalues, self.sigma))),
        )
