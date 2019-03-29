from autofit.mapper import model_mapper as mm
from autofit.mapper import prior_model as pm


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


class TestCase(object):
    def test_complex_class(self):
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

    def test_list_arguments(self):
        prior_model = pm.PriorModel(ListClass)

        assert prior_model.prior_count == 0

        prior_model = pm.PriorModel(ListClass, ls=[SimpleClass])

        assert prior_model.prior_count == 2

        prior_model = pm.PriorModel(ListClass, ls=[SimpleClass, SimpleClass])

        assert prior_model.prior_count == 4
