import copy

import pytest

import autofit as af
from test_autofit import mock


@pytest.fixture(name="instance_prior_model")
def make_instance_prior_model():
    instance = mock.SimpleClass(1.0, 2.0)
    return af.AbstractPriorModel.from_instance(instance)


@pytest.fixture(name="list_prior_model")
def make_list_prior_model():
    instance = [mock.SimpleClass(1.0, 2.0)]
    return af.AbstractPriorModel.from_instance(instance)


@pytest.fixture(name="complex_prior_model")
def make_complex_prior_model():
    instance = mock.ComplexClass(mock.SimpleClass(1.0, 2.0))
    return af.AbstractPriorModel.from_instance(instance)


class TestAsmodel:
    def test_instance(self, instance_prior_model):
        model = instance_prior_model.as_model()
        assert model.prior_count == 2

    def test_list(self, list_prior_model):
        model = list_prior_model.as_model()
        assert isinstance(model, af.CollectionPriorModel)
        assert model.prior_count == 2

    def test_complex(self, complex_prior_model):
        assert complex_prior_model.prior_count == 0
        model = complex_prior_model.as_model()
        assert model.prior_count == 2
        assert model.simple.prior_count == 2

    def test_galaxy_list(self):
        galaxies = af.ModelInstance()
        galaxies.one = mock.Galaxy()
        instance = af.ModelInstance()
        instance.galaxies = galaxies
        model = instance.as_model(model_classes=(mock.Galaxy,))
        assert model.prior_count == 1


class TestFromInstance:
    def test_model_mapper(self):
        instance = af.ModelInstance()
        instance.simple = mock.SimpleClass(1.0, 2.0)

        result = af.AbstractPriorModel.from_instance(instance)

        assert isinstance(result, af.ModelMapper)

    def test_with_model_classes(self):
        instance = mock.ComplexClass(mock.SimpleClass(1.0, 2.0))
        model = af.AbstractPriorModel.from_instance(
            instance, model_classes=(mock.SimpleClass,)
        )
        assert model.prior_count == 2

    def test_list_with_model_classes(self):
        instance = [
            mock.SimpleClass(1.0, 2.0),
            mock.ComplexClass(mock.SimpleClass(1.0, 2.0)),
        ]
        model = af.AbstractPriorModel.from_instance(
            instance, model_classes=(mock.ComplexClass,)
        )

        assert model.prior_count == 2
        assert model[0].prior_count == 0
        assert model[1].prior_count == 2

    def test_dict_with_model_classes(self):
        instance = {
            "one": mock.SimpleClass(1.0, 2.0),
            "two": mock.ComplexClass(mock.SimpleClass(1.0, 2.0)),
        }
        model = af.AbstractPriorModel.from_instance(
            instance, model_classes=(mock.ComplexClass,)
        )

        assert model.prior_count == 2
        assert model[0].prior_count == 0
        assert model[1].prior_count == 2

        assert model.one.one == 1.0
        assert model.one.two == 2.0

        assert isinstance(model.two.simple.one, af.Prior)
        assert isinstance(model.two.simple.two, af.Prior)

    def test_instance(self, instance_prior_model):
        assert instance_prior_model.cls == mock.SimpleClass
        assert instance_prior_model.prior_count == 0
        assert instance_prior_model.one == 1.0
        assert instance_prior_model.two == 2.0

        new_instance = instance_prior_model.instance_for_arguments({})
        assert isinstance(new_instance, mock.SimpleClass)
        assert new_instance.one == 1.0
        assert new_instance.two == 2.0

    def test_complex(self, complex_prior_model):
        assert complex_prior_model.cls == mock.ComplexClass
        assert complex_prior_model.prior_count == 0
        assert isinstance(complex_prior_model.simple, af.PriorModel)
        assert complex_prior_model.simple.cls == mock.SimpleClass
        assert complex_prior_model.simple.one == 1.0

        new_instance = complex_prior_model.instance_for_arguments({})
        assert isinstance(new_instance, mock.ComplexClass)
        assert isinstance(new_instance.simple, mock.SimpleClass)
        assert new_instance.simple.one == 1.0

    def test_list(self, list_prior_model):
        assert isinstance(list_prior_model, af.CollectionPriorModel)
        assert isinstance(list_prior_model[0], af.PriorModel)
        assert list_prior_model[0].one == 1.0

    def test_dict(self):
        instance = {"simple": mock.SimpleClass(1.0, 2.0)}
        prior_model = af.AbstractPriorModel.from_instance(instance)
        assert isinstance(prior_model, af.CollectionPriorModel)
        assert isinstance(prior_model.simple, af.PriorModel)
        assert prior_model.simple.one == 1.0

        new_instance = prior_model.instance_for_arguments({})
        assert isinstance(new_instance.simple, mock.SimpleClass)

        prior_model = af.AbstractPriorModel.from_instance(new_instance)
        assert isinstance(prior_model, af.CollectionPriorModel)
        assert isinstance(prior_model.simple, af.PriorModel)
        assert prior_model.simple.one == 1.0

    def test_dimension_types(self):
        instance = mock.DistanceClass(mock.Distance(1.0), mock.Distance(2.0))
        result = af.AbstractPriorModel.from_instance(
            instance, model_classes=(mock.DistanceClass,)
        )
        assert isinstance(result.first, af.PriorModel)

        new_instance = result.instance_from_unit_vector([0.1, 0.2])
        assert isinstance(new_instance, mock.DistanceClass)
        assert isinstance(new_instance.first, mock.Distance)
        assert new_instance.first == 0.1


class TestSum(object):
    def test_add_prior_models(self):
        profile_1 = af.PriorModel(mock.EllipticalLP)
        profile_2 = af.PriorModel(mock.EllipticalLP)

        profile_1.axis_ratio = 1.0
        profile_2.phi = 0.0

        result = profile_1 + profile_2

        assert isinstance(result, af.PriorModel)
        assert result.cls == mock.EllipticalLP
        assert isinstance(result.axis_ratio, af.Prior)
        assert isinstance(result.phi, af.Prior)

    def test_fail_for_mismatch(self):
        profile_1 = af.PriorModel(mock.EllipticalLP)
        profile_2 = af.PriorModel(mock.EllipticalMassProfile)

        with pytest.raises(TypeError):
            profile_1 + profile_2

    def test_add_children(self):
        galaxy_1 = af.PriorModel(
            mock.Galaxy,
            light_profiles=af.CollectionPriorModel(light_1=mock.EllipticalLP),
            mass_profiles=af.CollectionPriorModel(mass_1=mock.EllipticalMassProfile),
        )
        galaxy_2 = af.PriorModel(
            mock.Galaxy,
            light_profiles=af.CollectionPriorModel(light_2=mock.EllipticalLP),
            mass_profiles=af.CollectionPriorModel(mass_2=mock.EllipticalMassProfile),
        )

        result = galaxy_1 + galaxy_2

        assert result.light_profiles.light_1 == galaxy_1.light_profiles.light_1
        assert result.light_profiles.light_2 == galaxy_2.light_profiles.light_2

        assert result.mass_profiles.mass_1 == galaxy_1.mass_profiles.mass_1
        assert result.mass_profiles.mass_2 == galaxy_2.mass_profiles.mass_2

    def test_prior_model_override(self):
        galaxy_1 = af.PriorModel(
            mock.Galaxy,
            light_profiles=af.CollectionPriorModel(light=mock.EllipticalLP()),
            mass_profiles=af.CollectionPriorModel(mass=mock.EllipticalMassProfile),
        )
        galaxy_2 = af.PriorModel(
            mock.Galaxy,
            light_profiles=af.CollectionPriorModel(light=mock.EllipticalLP),
            mass_profiles=af.CollectionPriorModel(mass=mock.EllipticalMassProfile()),
        )

        result = galaxy_1 + galaxy_2

        assert result.mass_profiles.mass == galaxy_1.mass_profiles.mass
        assert result.light_profiles.light == galaxy_2.light_profiles.light


class TestFloatAnnotation(object):
    def test_distance_from_distance(self):
        original = mock.Distance(1.0)
        # noinspection PyTypeChecker
        distance = mock.DistanceClass(first=original, second=2.0)

        assert distance.first is original

    # noinspection PyTypeChecker
    def test_instantiate_distance(self):
        distance = mock.DistanceClass(first=1.0, second=2.0)

        assert distance.first == 1.0
        assert distance.second == 2.0

        assert isinstance(distance.first, mock.Distance)
        assert isinstance(distance.second, mock.Distance)

        distance = mock.DistanceClass(1.0, 2.0)

        assert distance.first == 1.0
        assert distance.second == 2.0

        assert isinstance(distance.first, mock.Distance)
        assert isinstance(distance.second, mock.Distance)

    def test_distance(self):
        mapper = af.ModelMapper()
        mapper.object = mock.DistanceClass

        assert mapper.prior_count == 2

        result = mapper.instance_from_unit_vector([0.5, 1.0])
        assert isinstance(result.object, mock.DistanceClass)
        assert result.object.first == 0.5
        assert result.object.second == 1.0

        assert isinstance(result.object.first, mock.Distance)

    def test_position(self):
        mapper = af.ModelMapper()
        mapper.object = mock.PositionClass

        assert mapper.prior_count == 2
        result = mapper.instance_from_unit_vector([0.5, 1.0])
        assert isinstance(result.object, mock.PositionClass)
        assert result.object.position[0] == 0.5
        assert result.object.position[1] == 1.0

        assert isinstance(result.object.position[0], mock.Distance)
        assert isinstance(result.object.position[1], mock.Distance)

    # noinspection PyUnresolvedReferences
    def test_prior_linking(self):
        mapper = af.ModelMapper()
        mapper.a = mock.SimpleClass
        mapper.b = mock.SimpleClass

        assert mapper.prior_count == 4

        mapper.a.one = mapper.b.one

        assert mapper.prior_count == 3

        mapper.a.two = mapper.b.two

        assert mapper.prior_count == 2

        mapper.a.one = mapper.a.two
        mapper.b.one = mapper.b.two

        assert mapper.prior_count == 1

    def test_prior_tuples(self):
        prior_model = af.PriorModel(mock.DistanceClass)

        assert prior_model.unique_prior_tuples[0].name == "first"
        assert prior_model.unique_prior_tuples[1].name == "second"


class TestHashing(object):
    def test_is_hashable(self):
        assert hash(af.AbstractPriorModel()) is not None
        assert hash(af.PriorModel(mock.SimpleClass)) is not None
        assert (
            hash(af.AnnotationPriorModel(mock.SimpleClass, mock.SimpleClass, "one"))
            is not None
        )

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
        prior_model = af.PriorModel(mock.ListClass)

        assert prior_model.prior_count == 0

        prior_model = af.PriorModel(mock.ListClass, ls=[mock.SimpleClass])

        assert prior_model.prior_count == 2

        prior_model = af.PriorModel(
            mock.ListClass, ls=[mock.SimpleClass, mock.SimpleClass]
        )

        assert prior_model.prior_count == 4

    def test_float_argument(self):
        prior = af.UniformPrior(0.5, 2.0)
        prior_model = af.PriorModel(mock.Galaxy, redshift=prior)

        assert prior_model.prior_count == 1
        assert prior_model.priors[0] is prior

        prior_model = af.PriorModel(mock.Galaxy, redshift=4.0)
        assert prior_model.prior_count == 0
        assert prior_model.redshift == 4.0

        instance = prior_model.instance_for_arguments({})
        assert instance.redshift == 4.0

    def test_model_argument(self):
        lens_galaxy = af.PriorModel(mock.Galaxy)
        source_galaxy = mock.Galaxy()
        tracer = af.PriorModel(
            mock.Tracer, lens_galaxy=lens_galaxy, source_galaxy=source_galaxy
        )

        assert tracer.lens_galaxy is lens_galaxy
        assert tracer.prior_count == 1

        deferred_instance = tracer.instance_for_arguments({lens_galaxy.redshift: 0.5})
        instance = deferred_instance(grid=None)

        assert instance.lens_galaxy.redshift == 0.5
        assert instance.source_galaxy is source_galaxy

    def test_no_passing(self):
        mapper = af.ModelMapper()
        mapper.distance = mock.DistanceClass
        instance = mapper.instance_from_prior_medians()
        assert not hasattr(instance.distance.first, "value") or not isinstance(
            instance.distance.first.value, af.Prior
        )

    def test_arbitrary_keyword_arguments(self):
        prior_model = af.PriorModel(
            mock.Galaxy,
            light=mock.EllipticalCoredIsothermal,
            mass=mock.EllipticalMassProfile,
        )
        assert prior_model.prior_count == 11
        instance = prior_model.instance_from_unit_vector(
            [0.5] * prior_model.prior_count
        )
        assert isinstance(instance.light, mock.EllipticalCoredIsothermal)
        assert isinstance(instance.mass, mock.EllipticalMassProfile)


class TestCase(object):
    def test_complex_class(self):
        prior_model = af.PriorModel(mock.ComplexClass)

        assert hasattr(prior_model, "simple")
        assert prior_model.simple.prior_count == 2
        assert prior_model.prior_count == 2

    def test_create_instance(self):
        mapper = af.ModelMapper()
        mapper.complex = mock.ComplexClass

        instance = mapper.instance_from_unit_vector([1.0, 0.0])

        assert instance.complex.simple.one == 1.0
        assert instance.complex.simple.two == 0.0

    def test_instantiate_with_list_arguments(self):
        mapper = af.ModelMapper()
        mapper.list_object = af.PriorModel(
            mock.ListClass, ls=[mock.SimpleClass, mock.SimpleClass]
        )

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
        mapper.list_object = af.PriorModel(
            mock.ListClass, ls=[mock.SimpleClass, mock.SimpleClass(1, 2)]
        )

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
            one=mock.SimpleClass, two=mock.SimpleClass(1, 2)
        )

        assert len(prior_model.direct_prior_model_tuples) == 1
        assert len(prior_model) == 2

        instance = prior_model.instance_for_arguments(
            {prior_model.one.one: 0.1, prior_model.one.two: 0.2}
        )

        assert instance.one.one == 0.1
        assert instance.one.two == 0.2

        assert instance.two.one == 1
        assert instance.two.two == 2

    def test_mix_instances_in_list_prior_model(self):
        prior_model = af.CollectionPriorModel(
            [mock.SimpleClass, mock.SimpleClass(1, 2)]
        )

        assert len(prior_model.direct_prior_model_tuples) == 1
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
        prior_model = af.CollectionPriorModel([[mock.SimpleClass]])

        assert len(prior_model.direct_prior_model_tuples) == 1
        assert prior_model.prior_count == 2

    def test_list_prior_model_with_dictionary(self, simple_model):
        assert isinstance(simple_model.simple, af.PriorModel)

    def test_override_with_instance(self, simple_model):
        simple_instance = mock.SimpleClass(1, 2)

        simple_model.simple = simple_instance

        assert len(simple_model) == 1
        assert simple_model.simple == simple_instance


@pytest.fixture(name="simple_model")
def make_simple_model():
    return af.CollectionPriorModel({"simple": mock.SimpleClass})


class TestCopy:
    def test_simple(self, simple_model):
        assert simple_model.prior_count > 0
        assert copy.deepcopy(simple_model).prior_count == simple_model.prior_count

    def test_embedded(self, simple_model):
        model = af.CollectionPriorModel(simple=simple_model)
        assert copy.deepcopy(model).prior_count == model.prior_count

    def test_circular(self):
        one = af.PriorModel(mock.SimpleClass)

        one.one = af.PriorModel(mock.SimpleClass)
        one.one.one = one

        # noinspection PyUnresolvedReferences
        assert one.prior_count == one.one.prior_count
        assert copy.deepcopy(one).prior_count == one.prior_count
