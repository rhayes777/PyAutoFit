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

    def test_instantiate_with_list_arguments(self):
        mapper = mm.ModelMapper()
        mapper.list_object = pm.PriorModel(ListClass, ls=[SimpleClass, SimpleClass])

        assert len(mapper.list_object.ls) == 2

        assert mapper.list_object.prior_count == 4
        assert mapper.prior_count == 4

        instance = mapper.instance_from_unit_vector([0.1, 0.2, 0.3, 0.4])

        assert len(instance.list_object.ls) == 2
        assert instance.list_object.ls[0].one == 0.1
        assert instance.list_object.ls[0].two == 0.2
        assert instance.list_object.ls[1].one == 0.3
        assert instance.list_object.ls[1].two == 0.4

    def test_mix_instances_and_models(self):
        mapper = mm.ModelMapper()
        mapper.list_object = pm.PriorModel(ListClass, ls=[SimpleClass, SimpleClass(1, 2)])

        assert mapper.prior_count == 2

        instance = mapper.instance_from_unit_vector([0.1, 0.2])

        assert len(instance.list_object.ls) == 2
        assert instance.list_object.ls[0].one == 0.1
        assert instance.list_object.ls[0].two == 0.2
        assert instance.list_object.ls[1].one == 1
        assert instance.list_object.ls[1].two == 2

    def test_mix_instances_in_list_prior_model(self):
        prior_model = pm.ListPriorModel([SimpleClass, SimpleClass(1, 2)])

        assert len(prior_model.prior_models) == 1
        assert prior_model.prior_count == 2

        mapper = mm.ModelMapper()
        mapper.ls = prior_model

        instance = mapper.instance_from_unit_vector([0.1, 0.2])

        assert len(instance.ls) == 2

        assert instance.ls[0].one == 0.1
        assert instance.ls[0].two == 0.2
        assert instance.ls[1].one == 1
        assert instance.ls[1].two == 2

    def test_list_in_list_prior_model(self):
        prior_model = pm.ListPriorModel([[SimpleClass]])

        assert len(prior_model.prior_models) == 1
        assert prior_model.prior_count == 2

    # def test_list_prior_model_with_dictionary(self):
    #     prior_model = pm.ListPriorModel({"simple": SimpleClass})
    #
    #     assert isinstance(prior_model.simple, pm.PriorModel)
