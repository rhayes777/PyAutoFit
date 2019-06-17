import pytest

from autofit.mapper import model
from autofit.mapper import model_mapper, prior as p
from test import mock
from test.mapper.test_model_mapper import MockClassMM, MockProfile


@pytest.fixture(name="galaxy_1")
def make_galaxy_1():
    return mock.Galaxy()


@pytest.fixture(name="galaxy_2")
def make_galaxy_2():
    return mock.Galaxy()


@pytest.fixture(name="instance")
def make_instance(galaxy_1, galaxy_2):
    sub = model.ModelInstance()

    instance = model.ModelInstance()
    sub.galaxy_1 = galaxy_1

    instance.galaxy_2 = galaxy_2
    instance.sub = sub

    sub_2 = model.ModelInstance()
    sub_2.galaxy_1 = galaxy_1

    instance.sub.sub = sub_2

    return instance


class TestModelInstance(object):
    def test_instances_of(self):
        instance = model.ModelInstance()
        instance.galaxy_1 = mock.Galaxy()
        instance.galaxy_2 = mock.Galaxy()
        assert instance.instances_of(mock.Galaxy) == [instance.galaxy_1,
                                                      instance.galaxy_2]

    def test_object_for_path(self, instance, galaxy_1, galaxy_2):
        assert instance.object_for_path(("galaxy_2",)) == galaxy_2
        assert instance.object_for_path(("sub", "galaxy_1")) == galaxy_1
        assert instance.object_for_path(("sub", "sub", "galaxy_1")) == galaxy_1

    def test_path_instance_tuples_for_class(self, instance, galaxy_1, galaxy_2):
        result = instance.path_instance_tuples_for_class(mock.Galaxy)
        assert result[0] == (("galaxy_2",), galaxy_2)
        assert result[1] == (("sub", "galaxy_1"), galaxy_1)
        assert result[2] == (("sub", "sub", "galaxy_1"), galaxy_1)

    def test_instances_of_filtering(self):
        instance = model.ModelInstance()
        instance.galaxy_1 = mock.Galaxy()
        instance.galaxy_2 = mock.Galaxy()
        instance.other = mock.GalaxyModel()
        assert instance.instances_of(mock.Galaxy) == [instance.galaxy_1,
                                                      instance.galaxy_2]

    def test_instances_from_list(self):
        instance = model.ModelInstance()
        galaxy_1 = mock.Galaxy()
        galaxy_2 = mock.Galaxy()
        instance.galaxies = [galaxy_1, galaxy_2]
        assert instance.instances_of(mock.Galaxy) == [galaxy_1, galaxy_2]

    def test_non_trivial_instances_of(self):
        instance = model.ModelInstance()
        galaxy_1 = mock.Galaxy(redshift=1)
        galaxy_2 = mock.Galaxy(redshift=2)
        instance.galaxies = [galaxy_1, galaxy_2, mock.GalaxyModel]
        instance.galaxy_3 = mock.Galaxy(redshift=3)
        instance.galaxy_prior = mock.GalaxyModel()

        assert instance.instances_of(mock.Galaxy) == [instance.galaxy_3, galaxy_1,
                                                      galaxy_2]

    def test_simple_model(self):
        mapper = model_mapper.ModelMapper()

        mapper.mock_class = MockClassMM

        model_map = mapper.instance_from_unit_vector([1., 1.])

        assert isinstance(model_map.mock_class, MockClassMM)
        assert model_map.mock_class.one == 1.
        assert model_map.mock_class.two == 1.

    def test_two_object_model(self):
        mapper = model_mapper.ModelMapper()

        mapper.mock_class_1 = MockClassMM
        mapper.mock_class_2 = MockClassMM

        model_map = mapper.instance_from_unit_vector([1., 0., 0., 1.])

        assert isinstance(model_map.mock_class_1, MockClassMM)
        assert isinstance(model_map.mock_class_2, MockClassMM)

        assert model_map.mock_class_1.one == 1.
        assert model_map.mock_class_1.two == 0.

        assert model_map.mock_class_2.one == 0.
        assert model_map.mock_class_2.two == 1.

    def test_swapped_prior_construction(self):
        mapper = model_mapper.ModelMapper()

        mapper.mock_class_1 = MockClassMM
        mapper.mock_class_2 = MockClassMM

        # noinspection PyUnresolvedReferences
        mapper.mock_class_2.one = mapper.mock_class_1.one

        model_map = mapper.instance_from_unit_vector([1., 0., 0.])

        assert isinstance(model_map.mock_class_1, MockClassMM)
        assert isinstance(model_map.mock_class_2, MockClassMM)

        assert model_map.mock_class_1.one == 1.
        assert model_map.mock_class_1.two == 0.

        assert model_map.mock_class_2.one == 1.
        assert model_map.mock_class_2.two == 0.

    def test_prior_replacement(self):
        mapper = model_mapper.ModelMapper()

        mapper.mock_class = MockClassMM

        mapper.mock_class.one = p.UniformPrior(100, 200)

        model_map = mapper.instance_from_unit_vector([0., 0.])

        assert model_map.mock_class.one == 100.

    def test_tuple_arg(self):
        mapper = model_mapper.ModelMapper()

        mapper.mock_profile = MockProfile

        model_map = mapper.instance_from_unit_vector([1., 0., 0.])

        assert model_map.mock_profile.centre == (1., 0.)
        assert model_map.mock_profile.intensity == 0.

    def test_modify_tuple(self):
        mapper = model_mapper.ModelMapper()

        mapper.mock_profile = MockProfile

        # noinspection PyUnresolvedReferences
        mapper.mock_profile.centre.centre_0 = p.UniformPrior(1., 10.)

        model_map = mapper.instance_from_unit_vector([1., 1., 1.])

        assert model_map.mock_profile.centre == (10., 1.)

    def test_match_tuple(self):
        mapper = model_mapper.ModelMapper()

        mapper.mock_profile = MockProfile

        # noinspection PyUnresolvedReferences
        mapper.mock_profile.centre.centre_1 = mapper.mock_profile.centre.centre_0

        model_map = mapper.instance_from_unit_vector([1., 0.])

        assert model_map.mock_profile.centre == (1., 1.)
        assert model_map.mock_profile.intensity == 0.
