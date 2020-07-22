import pytest

import autofit as af
from test_autofit import mock
from test_autofit import mock_real


@pytest.fixture(name="galaxy_1")
def make_galaxy_1():
    return mock_real.Galaxy()


@pytest.fixture(name="galaxy_2")
def make_galaxy_2():
    return mock_real.Galaxy()


@pytest.fixture(name="instance")
def make_instance(galaxy_1, galaxy_2):
    sub = af.ModelInstance()

    instance = af.ModelInstance()
    sub.galaxy_1 = galaxy_1

    instance.galaxy_2 = galaxy_2
    instance.sub = sub

    sub_2 = af.ModelInstance()
    sub_2.galaxy_1 = galaxy_1

    instance.sub.sub = sub_2

    return instance


class TestModelInstance:
    def test_as_model(self, instance):

        model = instance.as_model()
        assert isinstance(model, af.ModelMapper)
        assert isinstance(model.galaxy_2, af.PriorModel)
        assert model.galaxy_2.cls == mock_real.Galaxy

    def test_object_for_path(self, instance, galaxy_1, galaxy_2):

        assert instance.object_for_path(("galaxy_2",)) is galaxy_2
        assert instance.object_for_path(("sub", "galaxy_1")) is galaxy_1
        assert instance.object_for_path(("sub", "sub", "galaxy_1")) is galaxy_1
        setattr(instance.object_for_path(("galaxy_2",)), "galaxy", galaxy_1)
        assert galaxy_2.galaxy is galaxy_1

    def test_path_instance_tuples_for_class(self, instance, galaxy_1, galaxy_2):
        result = instance.path_instance_tuples_for_class(mock_real.Galaxy)
        assert result[0] == (("galaxy_2",), galaxy_2)
        assert result[1] == (("sub", "galaxy_1"), galaxy_1)
        assert result[2] == (("sub", "sub", "galaxy_1"), galaxy_1)

    def test_simple_model(self):
        mapper = af.ModelMapper()

        mapper.mock_class = mock.MockClassx2

        model_map = mapper.instance_from_unit_vector([1.0, 1.0])

        assert isinstance(model_map.mock_class, mock.MockClassx2)
        assert model_map.mock_class.one == 1.0
        assert model_map.mock_class.two == 2.0

    def test_two_object_model(self):
        mapper = af.ModelMapper()

        mapper.mock_class_1 = mock.MockClassx2
        mapper.mock_class_2 = mock.MockClassx2

        model_map = mapper.instance_from_unit_vector([1.0, 0.0, 0.0, 1.0])

        assert isinstance(model_map.mock_class_1, mock.MockClassx2)
        assert isinstance(model_map.mock_class_2, mock.MockClassx2)

        assert model_map.mock_class_1.one == 1.0
        assert model_map.mock_class_1.two == 0.0

        assert model_map.mock_class_2.one == 0.0
        assert model_map.mock_class_2.two == 2.0

    def test_swapped_prior_construction(self):
        mapper = af.ModelMapper()

        mapper.mock_class_1 = mock.MockClassx2
        mapper.mock_class_2 = mock.MockClassx2

        # noinspection PyUnresolvedReferences
        mapper.mock_class_2.one = mapper.mock_class_1.one

        model_map = mapper.instance_from_unit_vector([1.0, 0.0, 0.0])

        assert isinstance(model_map.mock_class_1, mock.MockClassx2)
        assert isinstance(model_map.mock_class_2, mock.MockClassx2)

        assert model_map.mock_class_1.one == 1.0
        assert model_map.mock_class_1.two == 0.0

        assert model_map.mock_class_2.one == 1.0
        assert model_map.mock_class_2.two == 0.0

    def test_prior_replacement(self):
        mapper = af.ModelMapper()

        mapper.mock_class = mock.MockClassx2

        mapper.mock_class.one = af.UniformPrior(100, 200)

        model_map = mapper.instance_from_unit_vector([0.0, 0.0])

        assert model_map.mock_class.one == 100.0

    def test_tuple_arg(self):
        mapper = af.ModelMapper()

        mapper.mock_profile = mock.MockClassx3TupleFloat

        model_map = mapper.instance_from_unit_vector([1.0, 0.0, 0.0])

        assert model_map.mock_profile.one_tuple == (1.0, 0.0)
        assert model_map.mock_profile.two == 0.0

    def test_modify_tuple(self):
        mapper = af.ModelMapper()

        mapper.mock_profile = mock.MockClassx3TupleFloat

        # noinspection PyUnresolvedReferences
        mapper.mock_profile.one_tuple.one_tuple_0 = af.UniformPrior(1.0, 10.0)

        model_map = mapper.instance_from_unit_vector([1.0, 1.0, 1.0])

        assert model_map.mock_profile.one_tuple == (10.0, 2.0)

    def test_match_tuple(self):
        mapper = af.ModelMapper()

        mapper.mock_profile = mock.MockClassx3TupleFloat

        # noinspection PyUnresolvedReferences
        mapper.mock_profile.one_tuple.one_tuple_1 = mapper.mock_profile.one_tuple.one_tuple_0

        model_map = mapper.instance_from_unit_vector([1.0, 0.0])

        assert model_map.mock_profile.one_tuple == (1.0, 1.0)
        assert model_map.mock_profile.two == 0.0
