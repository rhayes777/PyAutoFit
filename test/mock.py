import typing

from autofit.optimize import non_linear
from autofit.tools import dimension_type


class MockNonLinearOptimizer(non_linear.NonLinearOptimizer):

    def __init__(self, phase_name, phase_tag=None, phase_folders=None,
                 model_mapper=None,
                 most_probable=None, most_likely=None, model_upper_params=None,
                 model_lower_params=None):
        super(MockNonLinearOptimizer, self).__init__(phase_name=phase_name,
                                                     phase_tag=phase_tag,
                                                     phase_folders=phase_folders,
                                                     model_mapper=model_mapper)

        self.most_probable = most_probable
        self.most_likely = most_likely
        self.model_upper_params = model_upper_params
        self.model_lower_params = model_lower_params

    @property
    def most_probable_model_parameters(self):
        """
        Read the most probable or most likely model values from the 'obj_summary.txt' file which nlo from a \
        multinest lensing.

        This file stores the parameters of the most probable model in the first half of entries and the most likely
        model in the second half of entries. The offset parameter is used to start at the desired model.

        """
        return self.most_probable

    @property
    def most_likely_model_parameters(self):
        """
        Read the most probable or most likely model values from the 'obj_summary.txt' file which nlo from a \
        multinest lensing.

        This file stores the parameters of the most probable model in the first half of entries and the most likely
        model in the second half of entries. The offset parameter is used to start at the desired model.
        """
        return self.most_likely

    def model_parameters_at_upper_sigma_limit(self, sigma_limit):
        return self.model_upper_params

    def model_parameters_at_lower_sigma_limit(self, sigma_limit):
        return self.model_lower_params


class MockClassNLOx4(object):

    def __init__(self, one=1, two=2, three=3, four=4):
        self.one = one
        self.two = two
        self.three = three
        self.four = four


class MockClassNLOx5(object):

    def __init__(self, one=1, two=2, three=3, four=4, five=5):
        self.one = one
        self.two = two
        self.three = three
        self.four = four
        self.five = five


class MockClassNLOx6(object):

    def __init__(self, one=(1, 2), two=(3, 4), three=3, four=4):
        self.one = one
        self.two = two
        self.three = three
        self.four = four


class MockAnalysis(object):
    def __init__(self):
        self.kwargs = None
        self.instance = None
        self.visualise_instance = None

    def fit(self, instance):
        self.instance = instance
        return 1.

    # noinspection PyUnusedLocal
    def visualize(self, instance, *args, **kwargs):
        self.visualise_instance = instance


class SimpleClass(object):
    def __init__(self, one, two: float):
        self.one = one
        self.two = two


class ComplexClass(object):
    def __init__(self, simple: SimpleClass):
        self.simple = simple


class ListClass(object):
    def __init__(self, ls: list):
        self.ls = ls


class Distance(dimension_type.DimensionType):
    pass


class DistanceClass:
    @dimension_type.map_types
    def __init__(self, first: Distance, second: Distance):
        self.first = first
        self.second = second


class PositionClass:
    @dimension_type.map_types
    def __init__(self, position: typing.Tuple[Distance, Distance]):
        self.position = position
