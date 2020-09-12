import copy

import pytest

import autofit as af
from test_autofit import mock
from test_autofit import mock_real


@pytest.fixture(name="instance_prior_model")
def make_instance_prior_model():
    instance = mock.MockClassx2(1.0, 2.0)
    return af.AbstractPriorModel.from_instance(instance)


@pytest.fixture(name="list_prior_model")
def make_list_prior_model():
    instance = [mock.MockClassx2(1.0, 2.0)]
    return af.AbstractPriorModel.from_instance(instance)


@pytest.fixture(name="complex_prior_model")
def make_complex_prior_model():
    instance = mock.ComplexClass(mock.MockClassx2(1.0, 2.0))
    return af.AbstractPriorModel.from_instance(instance)


class TestAsModel:
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
        galaxies.one = mock.MockComponents()
        instance = af.ModelInstance()
        instance.galaxies = galaxies
        model = instance.as_model(model_classes=(mock.MockComponents,))
        assert model.prior_count == 1


class TestFromInstance:
    def test_model_mapper(self):
        instance = af.ModelInstance()
        instance.simple = mock.MockClassx2(1.0, 2.0)

        result = af.AbstractPriorModel.from_instance(instance)

        assert isinstance(result, af.ModelMapper)

    def test_with_model_classes(self):
        instance = mock.ComplexClass(mock.MockClassx2(1.0, 2.0))
        model = af.AbstractPriorModel.from_instance(
            instance, model_classes=(mock.MockClassx2,)
        )
        assert model.prior_count == 2

    def test_list_with_model_classes(self):
        instance = [
            mock.MockClassx2(1.0, 2.0),
            mock.ComplexClass(mock.MockClassx2(1.0, 2.0)),
        ]
        model = af.AbstractPriorModel.from_instance(
            instance, model_classes=(mock.ComplexClass,)
        )

        assert model.prior_count == 2
        assert model[0].prior_count == 0
        assert model[1].prior_count == 2

    def test_dict_with_model_classes(self):
        instance = {
            "one": mock.MockClassx2(1.0, 2.0),
            "two": mock.ComplexClass(mock.MockClassx2(1.0, 2.0)),
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
        assert instance_prior_model.cls == mock.MockClassx2
        assert instance_prior_model.prior_count == 0
        assert instance_prior_model.one == 1.0
        assert instance_prior_model.two == 2.0

        new_instance = instance_prior_model.instance_for_arguments({})
        assert isinstance(new_instance, mock.MockClassx2)
        assert new_instance.one == 1.0
        assert new_instance.two == 2.0

    def test_complex(self, complex_prior_model):
        assert complex_prior_model.cls == mock.ComplexClass
        assert complex_prior_model.prior_count == 0
        assert isinstance(complex_prior_model.simple, af.PriorModel)
        assert complex_prior_model.simple.cls == mock.MockClassx2
        assert complex_prior_model.simple.one == 1.0

        new_instance = complex_prior_model.instance_for_arguments({})
        assert isinstance(new_instance, mock.ComplexClass)
        assert isinstance(new_instance.simple, mock.MockClassx2)
        assert new_instance.simple.one == 1.0

    def test_list(self, list_prior_model):
        assert isinstance(list_prior_model, af.CollectionPriorModel)
        assert isinstance(list_prior_model[0], af.PriorModel)
        assert list_prior_model[0].one == 1.0

    def test_dict(self):
        instance = {"simple": mock.MockClassx2(1.0, 2.0)}
        prior_model = af.AbstractPriorModel.from_instance(instance)
        assert isinstance(prior_model, af.CollectionPriorModel)
        assert isinstance(prior_model.simple, af.PriorModel)
        assert prior_model.simple.one == 1.0

        new_instance = prior_model.instance_for_arguments({})
        assert isinstance(new_instance.simple, mock.MockClassx2)

        prior_model = af.AbstractPriorModel.from_instance(new_instance)
        assert isinstance(prior_model, af.CollectionPriorModel)
        assert isinstance(prior_model.simple, af.PriorModel)
        assert prior_model.simple.one == 1.0

    def test_dimension_types(self):
        instance = mock.MockDistanceClass(mock.MockDistance(1.0), mock.MockDistance(2.0))
        result = af.AbstractPriorModel.from_instance(
            instance, model_classes=(mock.MockDistanceClass,)
        )
        assert isinstance(result.one, af.PriorModel)

        new_instance = result.instance_from_unit_vector([0.1, 0.2])
        assert isinstance(new_instance, mock.MockDistanceClass)
        assert isinstance(new_instance.one, mock.MockDistance)
        assert new_instance.one == 0.1


class TestSum:
    def test_add_prior_models(self):
        profile_1 = af.PriorModel(mock_real.EllipticalProfile)
        profile_2 = af.PriorModel(mock_real.EllipticalProfile)

        profile_1.axis_ratio = 1.0
        profile_2.phi = 0.0

        result = profile_1 + profile_2

        assert isinstance(result, af.PriorModel)
        assert result.cls == mock_real.EllipticalProfile
        assert isinstance(result.axis_ratio, af.Prior)
        assert isinstance(result.phi, af.Prior)

    def test_fail_for_mismatch(self):
        profile_1 = af.PriorModel(mock_real.EllipticalProfile)
        profile_2 = af.PriorModel(mock_real.EllipticalMassProfile)

        with pytest.raises(TypeError):
            profile_1 + profile_2

    def test_add_children(self):
        galaxy_1 = af.PriorModel(
            mock.MockComponents,
            components_0=af.CollectionPriorModel(light_1=mock_real.EllipticalProfile),
            components_1=af.CollectionPriorModel(mass_1=mock_real.EllipticalMassProfile),
        )
        galaxy_2 = af.PriorModel(
            mock.MockComponents,
            components_0=af.CollectionPriorModel(light_2=mock_real.EllipticalProfile),
            components_1=af.CollectionPriorModel(mass_2=mock_real.EllipticalMassProfile),
        )

        result = galaxy_1 + galaxy_2

        assert result.components_0.light_1 == galaxy_1.components_0.light_1
        assert result.components_0.light_2 == galaxy_2.components_0.light_2

        assert result.components_1.mass_1 == galaxy_1.components_1.mass_1
        assert result.components_1.mass_2 == galaxy_2.components_1.mass_2

    def test_prior_model_override(self):
        galaxy_1 = af.PriorModel(
            mock.MockComponents,
            components_0=af.CollectionPriorModel(light=mock_real.EllipticalProfile()),
            components_1=af.CollectionPriorModel(mass=mock_real.EllipticalMassProfile),
        )
        galaxy_2 = af.PriorModel(
            mock.MockComponents,
            components_0=af.CollectionPriorModel(light=mock_real.EllipticalProfile),
            components_1=af.CollectionPriorModel(mass=mock_real.EllipticalMassProfile()),
        )

        result = galaxy_1 + galaxy_2

        assert result.components_1.mass == galaxy_1.components_1.mass
        assert result.components_0.light == galaxy_2.components_0.light


class TestFloatAnnotation:
    def test_distance_from_distance(self):
        original = mock.MockDistance(1.0)
        # noinspection PyTypeChecker
        distance = mock.MockDistanceClass(one=original, two=2.0)

        assert distance.one is original

    # noinspection PyTypeChecker
    def test_instantiate_distance(self):
        distance = mock.MockDistanceClass(one=1.0, two=2.0)

        assert distance.one == 1.0
        assert distance.two == 2.0

        assert isinstance(distance.one, mock.MockDistance)
        assert isinstance(distance.two, mock.MockDistance)

        distance = mock.MockDistanceClass(1.0, 2.0)

        assert distance.one == 1.0
        assert distance.two == 2.0

        assert isinstance(distance.one, mock.MockDistance)
        assert isinstance(distance.two, mock.MockDistance)

    def test_distance(self):
        mapper = af.ModelMapper()
        mapper.object = mock.MockDistanceClass

        assert mapper.prior_count == 2

        result = mapper.instance_from_unit_vector([0.5, 1.0])
        assert isinstance(result.object, mock.MockDistanceClass)
        assert result.object.one == 0.5
        assert result.object.two == 1.0

        assert isinstance(result.object.one, mock.MockDistance)

    def test_position(self):
        mapper = af.ModelMapper()
        mapper.object = mock.MockPositionClass

        assert mapper.prior_count == 2
        result = mapper.instance_from_unit_vector([0.5, 1.0])
        assert isinstance(result.object, mock.MockPositionClass)
        assert result.object.position[0] == 0.5
        assert result.object.position[1] == 1.0

        assert isinstance(result.object.position[0], mock.MockDistance)
        assert isinstance(result.object.position[1], mock.MockDistance)

    # noinspection PyUnresolvedReferences
    def test_prior_linking(self):
        mapper = af.ModelMapper()
        mapper.a = mock.MockClassx2
        mapper.b = mock.MockClassx2

        assert mapper.prior_count == 4

        mapper.a.one = mapper.b.one

        assert mapper.prior_count == 3

        mapper.a.two = mapper.b.two

        assert mapper.prior_count == 2

        mapper.a.one = mapper.a.two
        mapper.b.one = mapper.b.two

        assert mapper.prior_count == 1

    def test_prior_tuples(self):
        prior_model = af.PriorModel(mock.MockDistanceClass)

        assert prior_model.unique_prior_tuples[0].name == "one"
        assert prior_model.unique_prior_tuples[1].name == "two"


class TestHashing:
    def test_is_hashable(self):
        assert hash(af.AbstractPriorModel()) is not None
        assert hash(af.PriorModel(mock.MockClassx2)) is not None
        assert (
                hash(af.AnnotationPriorModel(mock.MockClassx2, mock.MockClassx2, "one"))
                is not None
        )

    def test_prior_prior_model_hash_consecutive(self):
        prior = af.UniformPrior(0, 1)
        prior_model = af.AbstractPriorModel()

        assert prior.id + 1 == prior_model.id


class StringDefault:
    def __init__(self, value="a string"):
        self.value = value


class TestStringArguments:
    def test_string_default(self):
        prior_model = af.PriorModel(StringDefault)
        assert prior_model.prior_count == 0

        assert prior_model.instance_for_arguments({}).value == "a string"


class TestPriorModelArguments:
    def test_list_arguments(self):
        prior_model = af.PriorModel(mock.ListClass)

        assert prior_model.prior_count == 0

        prior_model = af.PriorModel(mock.ListClass, ls=[mock.MockClassx2])

        assert prior_model.prior_count == 2

        prior_model = af.PriorModel(
            mock.ListClass, ls=[mock.MockClassx2, mock.MockClassx2]
        )

        assert prior_model.prior_count == 4

    def test_float_argument(self):
        prior = af.UniformPrior(0.5, 2.0)
        prior_model = af.PriorModel(mock.MockComponents, parameter=prior)

        assert prior_model.prior_count == 1
        assert prior_model.priors[0] is prior

        prior_model = af.PriorModel(mock.MockComponents, parameter=4.0)
        assert prior_model.prior_count == 0
        assert prior_model.parameter == 4.0

        instance = prior_model.instance_for_arguments({})
        assert instance.parameter == 4.0

    def test_no_passing(self):
        mapper = af.ModelMapper()
        mapper.distance = mock.MockDistanceClass
        instance = mapper.instance_from_prior_medians()
        assert not hasattr(instance.distance.one, "value") or not isinstance(
            instance.distance.one.value, af.Prior
        )

    def test_arbitrary_keyword_arguments(self):
        prior_model = af.PriorModel(
            mock.MockComponents,
            light=mock_real.EllipticalCoredIsothermal,
            mass=mock_real.EllipticalMassProfile,
        )
        assert prior_model.prior_count == 11
        instance = prior_model.instance_from_unit_vector(
            [0.5] * prior_model.prior_count
        )
        assert isinstance(instance.light, mock_real.EllipticalCoredIsothermal)
        assert isinstance(instance.mass, mock_real.EllipticalMassProfile)


class TestCase:
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
            mock.ListClass, ls=[mock.MockClassx2, mock.MockClassx2]
        )

        assert len(mapper.list_object.ls) == 2

        assert mapper.list_object.prior_count == 4
        assert mapper.prior_count == 4

        instance = mapper.instance_from_unit_vector([0.1, 0.2, 0.3, 0.4])

        assert len(instance.list_object.ls) == 2
        assert instance.list_object.ls[0].one == 0.1
        assert instance.list_object.ls[0].two == 0.4
        assert instance.list_object.ls[1].one == 0.3
        assert instance.list_object.ls[1].two == 0.8

    def test_mix_instances_and_models(self):
        mapper = af.ModelMapper()
        mapper.list_object = af.PriorModel(
            mock.ListClass, ls=[mock.MockClassx2, mock.MockClassx2(1, 2)]
        )

        assert mapper.prior_count == 2

        instance = mapper.instance_from_unit_vector([0.1, 0.2])

        assert len(instance.list_object.ls) == 2
        assert instance.list_object.ls[0].one == 0.1
        assert instance.list_object.ls[0].two == 0.4
        assert instance.list_object.ls[1].one == 1
        assert instance.list_object.ls[1].two == 2


class TestCollectionPriorModel:
    def test_keyword_arguments(self):
        prior_model = af.CollectionPriorModel(
            one=mock.MockClassx2, two=mock.MockClassx2(1, 2)
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
            [mock.MockClassx2, mock.MockClassx2(1, 2)]
        )

        assert len(prior_model.direct_prior_model_tuples) == 1
        assert prior_model.prior_count == 2

        mapper = af.ModelMapper()
        mapper.ls = prior_model

        instance = mapper.instance_from_unit_vector([0.1, 0.2])

        assert len(instance.ls) == 2

        assert instance.ls[0].one == 0.1
        assert instance.ls[0].two == 0.4
        assert instance.ls[1].one == 1
        assert instance.ls[1].two == 2

        assert len(prior_model.prior_class_dict) == 2

    def test_list_in_list_prior_model(self):
        prior_model = af.CollectionPriorModel([[mock.MockClassx2]])

        assert len(prior_model.direct_prior_model_tuples) == 1
        assert prior_model.prior_count == 2

    def test_list_prior_model_with_dictionary(self, simple_model):
        assert isinstance(simple_model.simple, af.PriorModel)

    def test_override_with_instance(self, simple_model):
        simple_instance = mock.MockClassx2(1, 2)

        simple_model.simple = simple_instance

        assert len(simple_model) == 1
        assert simple_model.simple == simple_instance

    def test_names_of_priors(self):
        collection = af.CollectionPriorModel([af.UniformPrior(), af.UniformPrior()])
        assert collection.name_for_prior(collection[0]) == "0"


@pytest.fixture(name="simple_model")
def make_simple_model():
    return af.CollectionPriorModel({"simple": mock.MockClassx2})


class TestCopy:
    def test_simple(self, simple_model):
        assert simple_model.prior_count > 0
        assert copy.deepcopy(simple_model).prior_count == simple_model.prior_count

    def test_embedded(self, simple_model):
        model = af.CollectionPriorModel(simple=simple_model)
        assert copy.deepcopy(model).prior_count == model.prior_count

    def test_circular(self):
        one = af.PriorModel(mock.MockClassx2)

        one.one = af.PriorModel(mock.MockClassx2)
        one.one.one = one

        # noinspection PyUnresolvedReferences
        assert one.prior_count == one.one.prior_count
        assert copy.deepcopy(one).prior_count == one.prior_count
