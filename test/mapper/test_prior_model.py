import pytest

import autofit as af
from test.mock import SimpleClass, ComplexClass, ListClass, Distance, \
    DistanceClass, PositionClass, Galaxy, Tracer, EllipticalLP, EllipticalMassProfile


class TestSum(object):
    def test_add_prior_models(self):
        profile_1 = af.PriorModel(EllipticalLP)
        profile_2 = af.PriorModel(EllipticalLP)

        profile_1.axis_ratio = 1.0
        profile_2.phi = 0.0

        result = profile_1 + profile_2

        assert isinstance(result, af.PriorModel)
        assert result.cls == EllipticalLP
        assert isinstance(result.axis_ratio, af.Prior)
        assert isinstance(result.phi, af.Prior)

    def test_fail_for_mismatch(self):
        profile_1 = af.PriorModel(EllipticalLP)
        profile_2 = af.PriorModel(EllipticalMassProfile)

        with pytest.raises(TypeError):
            profile_1 + profile_2

    def test_add_children(self):
        galaxy_1 = af.PriorModel(
            Galaxy,
            light_profiles=af.CollectionPriorModel(
                light_1=EllipticalLP
            ),
            mass_profiles=af.CollectionPriorModel(
                mass_1=EllipticalMassProfile
            )
        )
        galaxy_2 = af.PriorModel(
            Galaxy,
            light_profiles=af.CollectionPriorModel(
                light_2=EllipticalLP
            ),
            mass_profiles=af.CollectionPriorModel(
                mass_2=EllipticalMassProfile
            )
        )

        result = galaxy_1 + galaxy_2

        assert result.light_profiles.light_1 == galaxy_1.light_profiles.light_1
        assert result.light_profiles.light_2 == galaxy_2.light_profiles.light_2

        assert result.mass_profiles.mass_1 == galaxy_1.mass_profiles.mass_1
        assert result.mass_profiles.mass_2 == galaxy_2.mass_profiles.mass_2

    def test_prior_model_override(self):
        galaxy_1 = af.PriorModel(
            Galaxy,
            light_profiles=af.CollectionPriorModel(
                light=EllipticalLP()
            ),
            mass_profiles=af.CollectionPriorModel(
                mass=EllipticalMassProfile
            )
        )
        galaxy_2 = af.PriorModel(
            Galaxy,
            light_profiles=af.CollectionPriorModel(
                light=EllipticalLP
            ),
            mass_profiles=af.CollectionPriorModel(
                mass=EllipticalMassProfile()
            )
        )

        result = galaxy_1 + galaxy_2

        assert result.mass_profiles.mass == galaxy_1.mass_profiles.mass
        assert result.light_profiles.light == galaxy_2.light_profiles.light


class TestFloatAnnotation(object):
    def test_distance_from_distance(self):
        original = Distance(1.0)
        # noinspection PyTypeChecker
        distance = DistanceClass(first=original, second=2.0)

        assert distance.first is original

    # noinspection PyTypeChecker
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
        mapper = af.ModelMapper()
        mapper.object = DistanceClass

        assert mapper.prior_count == 2

        result = mapper.instance_from_unit_vector([0.5, 1.0])
        assert isinstance(result.object, DistanceClass)
        assert result.object.first == 0.5
        assert result.object.second == 1.0

        assert isinstance(result.object.first, Distance)

    def test_position(self):
        mapper = af.ModelMapper()
        mapper.object = PositionClass

        assert mapper.prior_count == 2
        result = mapper.instance_from_unit_vector([0.5, 1.0])
        assert isinstance(result.object, PositionClass)
        assert result.object.position[0] == 0.5
        assert result.object.position[1] == 1.0

        assert isinstance(result.object.position[0], Distance)
        assert isinstance(result.object.position[1], Distance)

    # noinspection PyUnresolvedReferences
    def test_prior_linking(self):
        mapper = af.ModelMapper()
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
        prior_model = af.PriorModel(DistanceClass)

        assert prior_model.prior_tuples[0].name == "first"
        assert prior_model.prior_tuples[1].name == "second"


class TestHashing(object):
    def test_is_hashable(self):
        assert hash(
            af.AbstractPriorModel()) is not None
        assert hash(af.PriorModel(SimpleClass)) is not None
        assert hash(
            af.AnnotationPriorModel(SimpleClass, SimpleClass, "one")) is not None

    def test_prior_prior_model_hash_consecutive(self):
        prior = af.Prior(0, 1)
        prior_model = af.AbstractPriorModel()

        assert prior.id + 1 == prior_model.id


class StringDefault:
    def __init__(self, value="a string"):
        self.value = value


class TestStringArguments(object):
    def test_string_default(self):
        prior_model = af.PriorModel(StringDefault)
        assert prior_model.prior_count == 0

        assert prior_model.instance_for_arguments({}).value == "a string"


class TestPriorModelArguments(object):
    def test_list_arguments(self):
        prior_model = af.PriorModel(ListClass)

        assert prior_model.prior_count == 0

        prior_model = af.PriorModel(ListClass, ls=[SimpleClass])

        assert prior_model.prior_count == 2

        prior_model = af.PriorModel(ListClass, ls=[SimpleClass, SimpleClass])

        assert prior_model.prior_count == 4

    def test_float_argument(self):
        prior = af.UniformPrior(0.5, 2.0)
        prior_model = af.PriorModel(Galaxy, redshift=prior)

        assert prior_model.prior_count == 1
        assert prior_model.priors[0] is prior

        prior_model = af.PriorModel(Galaxy, redshift=4.0)
        assert prior_model.prior_count == 0
        assert prior_model.redshift == 4.0

        instance = prior_model.instance_for_arguments({})
        assert instance.redshift == 4.0

    def test_model_argument(self):
        lens_galaxy = af.PriorModel(Galaxy)
        source_galaxy = Galaxy()
        tracer = af.PriorModel(
            Tracer,
            lens_galaxy=lens_galaxy,
            source_galaxy=source_galaxy
        )

        assert tracer.lens_galaxy is lens_galaxy
        assert tracer.prior_count == 1

        deferred_instance = tracer.instance_for_arguments(
            {
                lens_galaxy.redshift: 0.5
            }
        )
        instance = deferred_instance(grid=None)

        assert instance.lens_galaxy.redshift == 0.5
        assert instance.source_galaxy is source_galaxy


class TestCase(object):
    def test_complex_class(self):
        prior_model = af.PriorModel(ComplexClass)

        assert hasattr(prior_model, "simple")
        assert prior_model.simple.prior_count == 2
        assert prior_model.prior_count == 2

    def test_create_instance(self):
        mapper = af.ModelMapper()
        mapper.complex = ComplexClass

        instance = mapper.instance_from_unit_vector([1.0, 0.0])

        assert instance.complex.simple.one == 1.0
        assert instance.complex.simple.two == 0.0

    def test_instantiate_with_list_arguments(self):
        mapper = af.ModelMapper()
        mapper.list_object = af.PriorModel(ListClass, ls=[SimpleClass, SimpleClass])

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
        mapper = af.ModelMapper()
        mapper.list_object = af.PriorModel(ListClass,
                                           ls=[SimpleClass, SimpleClass(1, 2)])

        assert mapper.prior_count == 2

        instance = mapper.instance_from_unit_vector([0.1, 0.2])

        assert len(instance.list_object.ls) == 2
        assert instance.list_object.ls[0].one == 0.1
        assert instance.list_object.ls[0].two == 0.2
        assert instance.list_object.ls[1].one == 1
        assert instance.list_object.ls[1].two == 2


class TestCollectionPriorModel(object):
    def test_keyword_arguments(self):
        prior_model = af.CollectionPriorModel(
            one=SimpleClass,
            two=SimpleClass(1, 2)
        )

        assert len(prior_model.prior_models) == 1
        assert len(prior_model) == 2

        instance = prior_model.instance_for_arguments(
            {
                prior_model.one.one: 0.1,
                prior_model.one.two: 0.2
            }
        )

        assert instance.one.one == 0.1
        assert instance.one.two == 0.2

        assert instance.two.one == 1
        assert instance.two.two == 2

    def test_mix_instances_in_list_prior_model(self):
        prior_model = af.CollectionPriorModel([SimpleClass, SimpleClass(1, 2)])

        assert len(prior_model.prior_models) == 1
        assert prior_model.prior_count == 2

        mapper = af.ModelMapper()
        mapper.ls = prior_model

        instance = mapper.instance_from_unit_vector([0.1, 0.2])

        assert len(instance.ls) == 2

        assert instance.ls[0].one == 0.1
        assert instance.ls[0].two == 0.2
        assert instance.ls[1].one == 1
        assert instance.ls[1].two == 2

        assert len(prior_model.prior_class_dict) == 2

    def test_list_in_list_prior_model(self):
        prior_model = af.CollectionPriorModel([[SimpleClass]])

        assert len(prior_model.prior_models) == 1
        assert prior_model.prior_count == 2

    def test_list_prior_model_with_dictionary(self):
        prior_model = af.CollectionPriorModel({"simple": SimpleClass})

        assert isinstance(prior_model.simple, af.PriorModel)

    # def test_labels(self):
    #     mapper = af.ModelMapper()
    #
    #     mapper.my_list = af.CollectionPriorModel({"simple": SimpleClass})
    #
    #     assert mapper.info.split("\n")[4].startswith("my_list_simple_one")

    def test_override_with_constant(self):
        prior_model = af.CollectionPriorModel({"simple": SimpleClass})

        simple_instance = SimpleClass(1, 2)

        prior_model.simple = simple_instance

        assert len(prior_model) == 1
        assert prior_model.simple == simple_instance
