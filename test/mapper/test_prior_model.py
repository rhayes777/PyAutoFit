import inspect

from autofit.mapper import model_mapper as mm
from autofit.mapper import prior_model as pm


class SimpleClass(object):
    def __init__(self, one, two: float):
        self.one = one
        self.two = two


class ComplexClass(object):
    def __init__(self, simple: SimpleClass):
        self.simple = simple


class TestCase(object):
    def test_complex_class(self):
        arg_spec = inspect.getfullargspec(ComplexClass.__init__)
        print(arg_spec)
        arg_spec = inspect.getfullargspec(SimpleClass.__init__)
        print(arg_spec)

        prior_model = pm.PriorModel(ComplexClass)

        assert hasattr(prior_model, "simple")
        assert prior_model.simple.prior_count == 2
        assert prior_model.prior_count == 2

    def test_create_instance(self):
        mapper = mm.ModelMapper()
        mapper.complex = ComplexClass

        instance = mapper.instance_from_unit_vector([1.0, 0.0])

        assert instance.complex.simple.one == 1.0
        assert instance.complex.simple.two == 0.0
