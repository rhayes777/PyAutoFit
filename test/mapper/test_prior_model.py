import typing

from autofit.mapper import model_mapper as mm
from autofit.mapper import prior_model as pm
from autofit.tools import dimension_type


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


class TestFloatAnnotation(object):
    def test_distance_from_distance(self):
        original = Distance(1.0)
        distance = DistanceClass(first=original, second=2.0)

        assert distance.first is original

    def test_instantiate_distance(self):
        distance = DistanceClass(first=1.0, second=2.0)

        assert distance.first == 1.0
        assert distance.second == 2.0

        assert isinstance(distance.first, Distance)
        assert isinstance(distance.second, Distance)

        distance = DistanceClass(1.0, 2.0)

        assert distance.first == 1.0
        assert distance.second == 2.0

        assert isinstance(distance.first, Distance)
        assert isinstance(distance.second, Distance)

    def test_distance(self):
        mapper = mm.ModelMapper()
        mapper.object = DistanceClass

        assert mapper.prior_count == 2

        result = mapper.instance_from_unit_vector([0.5, 1.0])
        assert isinstance(result.object, DistanceClass)
        assert result.object.first == 0.5
        assert result.object.second == 1.0

        assert isinstance(result.object.first, Distance)

    def test_position(self):
        mapper = mm.ModelMapper()
        mapper.object = PositionClass

        assert mapper.prior_count == 2
        result = mapper.instance_from_unit_vector([0.5, 1.0])
        assert isinstance(result.object, PositionClass)
        assert result.object.position[0] == 0.5
        assert result.object.position[1] == 1.0

        assert isinstance(result.object.position[0], Distance)
        assert isinstance(result.object.position[1], Distance)

    def test_prior_linking(self):
        mapper = mm.ModelMapper()
        mapper.a = SimpleClass
        mapper.b = SimpleClass

        assert mapper.prior_count == 4

        mapper.a.one = mapper.b.one

        assert mapper.prior_count == 3

        mapper.a.two = mapper.b.two

        assert mapper.prior_count == 2

        mapper.a.one = mapper.a.two
        mapper.b.one = mapper.b.two

        assert mapper.prior_count == 1

    def test_prior_tuples(self):
        prior_model = pm.PriorModel(DistanceClass)

        assert prior_model.prior_tuples[0].name == "first"
        assert prior_model.prior_tuples[1].name == "second"


class TestHashing(object):
    def test_is_hashable(self):
        assert hash(pm.AbstractPriorModel()) is not None
        assert hash(pm.PriorModel(SimpleClass)) is not None
        assert hash(pm.AnnotationPriorModel(SimpleClass, SimpleClass, "one")) is not None

    def test_prior_prior_model_hash_consecutive(self):
        prior = pm.Prior(0, 1)
        prior_model = pm.AbstractPriorModel()

        assert prior.id + 1 == prior_model.id


class StringDefault:
    def __init__(self, value="a string"):
        self.value = value


class TestStringArguments(object):
    def test_string_default(self):
        prior_model = pm.PriorModel(StringDefault)
        assert prior_model.prior_count == 0

        assert prior_model.instance_for_arguments({}).value == "a string"


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
        prior_model = pm.CollectionPriorModel([SimpleClass, SimpleClass(1, 2)])

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

        assert len(prior_model.prior_class_dict) == 2

    def test_list_in_list_prior_model(self):
        prior_model = pm.CollectionPriorModel([[SimpleClass]])

        assert len(prior_model.prior_models) == 1
        assert prior_model.prior_count == 2

    def test_list_prior_model_with_dictionary(self):
        prior_model = pm.CollectionPriorModel({"simple": SimpleClass})

        assert isinstance(prior_model.simple, pm.PriorModel)

    def test_labels(self):
        mapper = mm.ModelMapper()

        mapper.my_list = pm.CollectionPriorModel({"simple": SimpleClass})

        assert mapper.info.split("\n")[4].startswith("my_list_simple_one")

    def test_override_with_constant(self):
        prior_model = pm.CollectionPriorModel({"simple": SimpleClass})

        simple_instance = SimpleClass(1, 2)

        prior_model.simple = simple_instance

        assert len(prior_model) == 1
        assert prior_model.simple == simple_instance
