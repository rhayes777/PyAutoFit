import os

import numpy as np
import pytest

import autofit as af
from autofit import exc
from autofit.text import formatter as frm
from test_autofit import mock
from test_autofit import mock_real

dataset_path = "{}/../".format(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture(name="initial_model")
def make_initial_model():
    return af.PriorModel(mock.MockClassx2)


class TestParamNames:
    def test_has_prior(self):
        prior_model = af.PriorModel(mock.MockClassx2)
        assert "one" == prior_model.name_for_prior(prior_model.one)


class MockClassx2instance(mock.MockClassx2):
    pass


class ExtendedMockClass(mock.MockClassx2):
    def __init__(self, one, two, three):
        super().__init__(one, two)
        self.three = three


@pytest.fixture(name="formatter")
def make_info_dict():
    formatter = frm.TextFormatter(line_length=20, indent=4)
    formatter.add((("one", "one"), 1))
    formatter.add((("one", "two"), 2))
    formatter.add((("one", "three", "four"), 4))
    formatter.add((("three", "four"), 4))

    return formatter


class TestGenerateModelInfo:
    def test_add_to_info_dict(self, formatter):
        print(formatter.dict)
        assert formatter.dict == {
            "one": {"one": 1, "two": 2, "three": {"four": 4}},
            "three": {"four": 4},
        }

    def test_info_string(self, formatter):
        ls = formatter.list

        assert ls[0] == "one"
        assert len(ls[1]) == 21
        assert ls[1] == "    one             1"
        assert ls[2] == "    two             2"
        assert ls[3] == "    three"
        assert ls[4] == "        four        4"
        assert ls[5] == "three"
        assert ls[6] == "    four            4"

    def test_basic(self):
        mm = af.ModelMapper()
        mm.mock_class = mock.MockClassx2
        model_info = mm.info

        assert (
                model_info
                == """mock_class
    one                                                                                   UniformPrior, lower_limit = 0.0, upper_limit = 1.0
    two                                                                                   UniformPrior, lower_limit = 0.0, upper_limit = 2.0"""
        )

    def test_with_instance(self):
        mm = af.ModelMapper()
        mm.mock_class = mock.MockClassx2

        mm.mock_class.two = 1.0

        model_info = mm.info

        assert (
                model_info
                == """mock_class
    one                                                                                   UniformPrior, lower_limit = 0.0, upper_limit = 1.0
    two                                                                                   1.0"""
        )

    def test_with_promise(self):
        mm = af.ModelMapper()
        mm.promise = af.prior.Promise(
            af.Phase(
                phase_name="phase",
                analysis_class=None,
                search=af.MockSearch()
            ),
            "path",
            result_path=[],
            assert_exists=False
        )

        assert mm.info == 'promise                                                                                   phase.result.model.path'

    def test_with_tuple(self):
        mm = af.ModelMapper()
        mm.tuple = (0, 1)

        assert mm.info == 'tuple                                                                                     (0, 1)'


class WithFloat:
    def __init__(self, value):
        self.value = value


class WithTuple:
    def __init__(self, tup=(0.0, 0.0)):
        self.tup = tup


# noinspection PyUnresolvedReferences
class TestRegression:
    def test_tuple_instance_model_info(self, mapper):
        mapper.profile = mock_real.EllipticalCoreSersic
        info = mapper.info

        mapper.profile.centre_0 = 1.0

        assert len(mapper.profile.centre.instance_tuples) == 1
        assert len(mapper.profile.instance_tuples) == 1

        assert len(info.split("\n")) == len(mapper.info.split("\n"))

    def test_set_tuple_instance(self):
        mm = af.ModelMapper()
        mm.sersic = mock_real.EllipticalSersic

        assert mm.prior_count == 7

        mm.sersic.centre_0 = 0.0
        mm.sersic.centre_1 = 0.0

        assert mm.prior_count == 5

    def test_get_tuple_instances(self):
        mm = af.ModelMapper()
        mm.sersic = mock_real.EllipticalSersic

        assert isinstance(mm.sersic.centre_0, af.Prior)
        assert isinstance(mm.sersic.centre_1, af.Prior)

    def test_tuple_parameter(self, mapper):
        mapper.with_float = WithFloat
        mapper.with_tuple = WithTuple

        assert mapper.prior_count == 3

        mapper.with_tuple.tup_0 = mapper.with_float.value

        assert mapper.prior_count == 2

    def test_parameter_name_ordering(self):
        mm = af.ModelMapper()
        mm.one = mock.MockClassRelativeWidth
        mm.two = mock.MockClassRelativeWidth

        mm.one.one.id = mm.two.three.id + 1

        assert mm.parameter_names == [
            "one_two",
            "one_three",
            "two_one",
            "two_two",
            "two_three",
            "one_one",
        ]

    def test_parameter_name_distinction(self):
        mm = af.ModelMapper()
        mm.ls = af.CollectionPriorModel(
            [
                af.PriorModel(mock.MockClassRelativeWidth),
                af.PriorModel(mock.MockClassRelativeWidth),
            ]
        )
        assert mm.parameter_names == [
            "ls_0_one",
            "ls_0_two",
            "ls_0_three",
            "ls_1_one",
            "ls_1_two",
            "ls_1_three",
        ]

    def test_name_for_prior(self):
        ls = af.CollectionPriorModel(
            [
                mock.MockClassRelativeWidth(1, 2, 3),
                af.PriorModel(mock.MockClassRelativeWidth),
            ]
        )
        assert ls.name_for_prior(ls[1].one) == "1_one"

    def test_tuple_parameter_float(self, mapper):
        mapper.with_float = WithFloat
        mapper.with_tuple = WithTuple

        mapper.with_float.value = 1.0

        assert mapper.prior_count == 2

        mapper.with_tuple.tup_0 = mapper.with_float.value

        assert mapper.prior_count == 1

        instance = mapper.instance_from_unit_vector([0.0])

        assert instance.with_float.value == 1
        assert instance.with_tuple.tup == (1.0, 0.0)


class TestModelingMapper:
    def test__argument_extraction(self):
        mapper = af.ModelMapper()
        mapper.mock_class = mock.MockClassx2
        assert 1 == len(mapper.prior_model_tuples)

        assert len(mapper.prior_tuples_ordered_by_id) == 2

    def test_attribution(self):
        mapper = af.ModelMapper()

        mapper.mock_class = mock.MockClassx2

        assert hasattr(mapper, "mock_class")
        assert hasattr(mapper.mock_class, "one")

    def test_tuple_arg(self):
        mapper = af.ModelMapper()

        mapper.mock_profile = mock.MockClassx3TupleFloat

        assert 3 == len(mapper.prior_tuples_ordered_by_id)


class TestRealClasses:
    def test_combination(self):
        mapper = af.ModelMapper(
            source_light_profile=mock_real.EllipticalSersic,
            lens_mass_profile=mock_real.EllipticalCoredIsothermal,
            lens_light_profile=mock_real.EllipticalCoreSersic,
        )

        model_map = mapper.instance_from_unit_vector(
            [1 for _ in range(len(mapper.prior_tuples_ordered_by_id))]
        )

        assert isinstance(
            model_map.source_light_profile, mock_real.EllipticalSersic
        )
        assert isinstance(
            model_map.lens_mass_profile, mock_real.EllipticalCoredIsothermal
        )
        assert isinstance(
            model_map.lens_light_profile, mock_real.EllipticalCoreSersic
        )

    def test_attribute(self):
        mm = af.ModelMapper()
        mm.cls_1 = mock.MockClassx2

        assert 1 == len(mm.prior_model_tuples)
        assert isinstance(mm.cls_1, af.PriorModel)


class TestConfigFunctions:
    def test_model_from_unit_vector(self):
        mapper = af.ModelMapper(geometry_profile=mock.MockClassx2Tuple)

        model_map = mapper.instance_from_unit_vector([1.0, 1.0])

        assert model_map.geometry_profile.one_tuple == (1.0, 2.0)

    def test_model_from_vector(self):
        mapper = af.ModelMapper(geometry_profile=mock.MockClassx2Tuple)

        model_map = mapper.instance_from_vector([1.0, 0.5])

        assert model_map.geometry_profile.one_tuple == (1.0, 0.5)

    def test_inheritance(self):
        mapper = af.ModelMapper(geometry_profile=mock_real.EllipticalProfile)

        model_map = mapper.instance_from_unit_vector([1.0, 1.0, 1.0, 1.0])

        assert model_map.geometry_profile.centre == (1.0, 1.0)

    def test_true_config(self):
        mapper = af.ModelMapper(
            sersic_light_profile=mock_real.EllipticalSersic,
            elliptical_profile_1=mock_real.EllipticalProfile,
            elliptical_profile_2=mock_real.EllipticalProfile,
            spherical_profile=mock_real.SphericalProfile,
            exponential_light_profile=mock_real.EllipticalExponential,
        )

        model_map = mapper.instance_from_unit_vector(
            [0.5 for _ in range(len(mapper.prior_tuples_ordered_by_id))]
        )

        assert isinstance(
            model_map.elliptical_profile_1, mock_real.EllipticalProfile
        )
        assert isinstance(
            model_map.elliptical_profile_2, mock_real.EllipticalProfile
        )
        assert isinstance(
            model_map.spherical_profile, mock_real.SphericalProfile
        )

        assert isinstance(
            model_map.sersic_light_profile, mock_real.EllipticalSersic
        )
        assert isinstance(
            model_map.exponential_light_profile, mock_real.EllipticalExponential
        )


class TestModelInstancesRealClasses:
    def test__in_order_of_class_constructor__one_profile(self):
        mapper = af.ModelMapper(profile_1=mock_real.EllipticalProfile)

        model_map = mapper.instance_from_unit_vector([0.25, 0.5, 0.75, 1.0])

        assert model_map.profile_1.centre == (0.25, 0.5)
        assert model_map.profile_1.axis_ratio == 1.5
        assert model_map.profile_1.phi == 2.0

    def test__in_order_of_class_constructor___multiple_profiles(self):
        mapper = af.ModelMapper(
            profile_1=mock_real.EllipticalProfile,
            profile_2=mock_real.SphericalProfile,
            profile_3=mock_real.EllipticalProfile,
        )

        model_map = mapper.instance_from_unit_vector(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )

        assert model_map.profile_1.centre == (0.1, 0.2)
        assert model_map.profile_1.axis_ratio == 0.6
        assert model_map.profile_1.phi == 0.8

        assert model_map.profile_2.centre == (0.5, 0.6)

        assert model_map.profile_3.centre == (0.7, 0.8)
        assert model_map.profile_3.axis_ratio == 1.8
        assert model_map.profile_3.phi == 2.0

    def test__check_order_for_different_unit_values(self):
        mapper = af.ModelMapper(
            profile_1=mock_real.EllipticalProfile,
            profile_2=mock_real.SphericalProfile,
            profile_3=mock_real.EllipticalProfile,
        )

        mapper.profile_1.centre.centre_0 = af.UniformPrior(0.0, 1.0)
        mapper.profile_1.centre.centre_1 = af.UniformPrior(0.0, 1.0)
        mapper.profile_1.axis_ratio = af.UniformPrior(0.0, 1.0)
        mapper.profile_1.phi = af.UniformPrior(0.0, 1.0)

        mapper.profile_2.centre.centre_0 = af.UniformPrior(0.0, 1.0)
        mapper.profile_2.centre.centre_1 = af.UniformPrior(0.0, 1.0)

        mapper.profile_3.centre.centre_0 = af.UniformPrior(0.0, 1.0)
        mapper.profile_3.centre.centre_1 = af.UniformPrior(0.0, 1.0)
        mapper.profile_3.axis_ratio = af.UniformPrior(0.0, 1.0)
        mapper.profile_3.phi = af.UniformPrior(0.0, 1.0)

        model_map = mapper.instance_from_unit_vector(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )

        assert model_map.profile_1.centre == (0.1, 0.2)
        assert model_map.profile_1.axis_ratio == 0.3
        assert model_map.profile_1.phi == 0.4

        assert model_map.profile_2.centre == (0.5, 0.6)

        assert model_map.profile_3.centre == (0.7, 0.8)
        assert model_map.profile_3.axis_ratio == 0.9
        assert model_map.profile_3.phi == 1.0

    def test__check_order_for_different_unit_values_and_set_priors_equal_to_one_another(
            self
    ):
        mapper = af.ModelMapper(
            profile_1=mock_real.EllipticalProfile,
            profile_2=mock_real.SphericalProfile,
            profile_3=mock_real.EllipticalProfile,
        )

        mapper.profile_1.centre.centre_0 = af.UniformPrior(0.0, 1.0)
        mapper.profile_1.centre.centre_1 = af.UniformPrior(0.0, 1.0)
        mapper.profile_1.axis_ratio = af.UniformPrior(0.0, 1.0)
        mapper.profile_1.phi = af.UniformPrior(0.0, 1.0)

        mapper.profile_2.centre.centre_0 = af.UniformPrior(0.0, 1.0)
        mapper.profile_2.centre.centre_1 = af.UniformPrior(0.0, 1.0)

        mapper.profile_3.centre.centre_0 = af.UniformPrior(0.0, 1.0)
        mapper.profile_3.centre.centre_1 = af.UniformPrior(0.0, 1.0)
        mapper.profile_3.axis_ratio = af.UniformPrior(0.0, 1.0)
        mapper.profile_3.phi = af.UniformPrior(0.0, 1.0)

        mapper.profile_1.axis_ratio = mapper.profile_1.phi
        mapper.profile_3.centre.centre_1 = mapper.profile_2.centre.centre_1

        model_map = mapper.instance_from_unit_vector(
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )

        assert model_map.profile_1.centre == (0.2, 0.3)
        assert model_map.profile_1.axis_ratio == 0.4
        assert model_map.profile_1.phi == 0.4

        assert model_map.profile_2.centre == (0.5, 0.6)

        assert model_map.profile_3.centre == (0.7, 0.6)
        assert model_map.profile_3.axis_ratio == 0.8
        assert model_map.profile_3.phi == 0.9

    def test__check_order_for_physical_values(self):
        mapper = af.ModelMapper(
            profile_1=mock_real.EllipticalProfile,
            profile_2=mock_real.SphericalProfile,
            profile_3=mock_real.EllipticalProfile,
        )

        model_map = mapper.instance_from_vector(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )

        assert model_map.profile_1.centre == (0.1, 0.2)
        assert model_map.profile_1.axis_ratio == 0.3
        assert model_map.profile_1.phi == 0.4

        assert model_map.profile_2.centre == (0.5, 0.6)

        assert model_map.profile_3.centre == (0.7, 0.8)
        assert model_map.profile_3.axis_ratio == 0.9
        assert model_map.profile_3.phi == 1.0

    def test__from_prior_medians__one_model(self):
        mapper = af.ModelMapper(profile_1=mock_real.EllipticalProfile)

        model_map = mapper.instance_from_prior_medians()

        model_2 = mapper.instance_from_unit_vector([0.5, 0.5, 0.5, 0.5])

        assert model_map.profile_1.centre == model_2.profile_1.centre == (0.5, 0.5)
        assert model_map.profile_1.axis_ratio == model_2.profile_1.axis_ratio == 1.0
        assert model_map.profile_1.phi == model_2.profile_1.phi == 1.0

    def test__from_prior_medians__multiple_models(self):
        mapper = af.ModelMapper(
            profile_1=mock_real.EllipticalProfile,
            profile_2=mock_real.SphericalProfile,
            profile_3=mock_real.EllipticalProfile,
        )

        model_map = mapper.instance_from_prior_medians()

        model_2 = mapper.instance_from_unit_vector(
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        )

        assert model_map.profile_1.centre == model_2.profile_1.centre == (0.5, 0.5)
        assert model_map.profile_1.axis_ratio == model_2.profile_1.axis_ratio == 1.0
        assert model_map.profile_1.phi == model_2.profile_1.phi == 1.0

        assert model_map.profile_2.centre == model_2.profile_2.centre == (0.5, 0.5)

        assert model_map.profile_3.centre == model_2.profile_3.centre == (0.5, 0.5)
        assert model_map.profile_3.axis_ratio == model_2.profile_3.axis_ratio == 1.0
        assert model_map.profile_3.phi == model_2.profile_3.phi == 1.0

    def test__from_prior_medians__one_model__set_one_parameter_to_another(self):
        mapper = af.ModelMapper(profile_1=mock_real.EllipticalProfile)

        mapper.profile_1.axis_ratio = mapper.profile_1.phi

        model_map = mapper.instance_from_prior_medians()

        model_2 = mapper.instance_from_unit_vector([0.5, 0.5, 0.5])

        assert model_map.profile_1.centre == model_2.profile_1.centre == (0.5, 0.5)
        assert model_map.profile_1.axis_ratio == model_2.profile_1.axis_ratio == 1.0
        assert model_map.profile_1.phi == model_2.profile_1.phi == 1.0

    def test_log_priors_from_vector(self):
        mapper = af.ModelMapper()
        mapper.mock_class = af.PriorModel(mock.MockClassx2)
        mapper.mock_class.one = af.GaussianPrior(mean=1.0, sigma=2.0)
        mapper.mock_class.two = af.LogUniformPrior(lower_limit=0.0, upper_limit=10.0)

        log_priors = mapper.log_priors_from_vector(vector=[0.0, 5.0])

        assert log_priors == [0.125, 0.2]

    def test_random_unit_vector_within_limits(self):

        mapper = af.ModelMapper()
        mapper.mock_class = af.PriorModel(mock.MockClassx2)

        np.random.seed(1)

        assert mapper.random_unit_vector_within_limits(lower_limit=0.0, upper_limit=1.0) == pytest.approx(
            [0.41702, 0.720324], 1.0e-4
        )

        assert mapper.random_unit_vector_within_limits(lower_limit=0.2, upper_limit=0.8) == pytest.approx(
            [0.200068, 0.38140], 1.0e-4
        )

    def test_random_vector_from_prior_within_limits(self):
        np.random.seed(1)

        mapper = af.ModelMapper()
        mapper.mock_class = af.PriorModel(mock.MockClassx2)

        vector = mapper.random_vector_from_priors_within_limits(lower_limit=0.499999, upper_limit=0.500001)

        assert vector == pytest.approx([0.5, 1.0], 1.0e-4)

        vector = mapper.random_vector_from_priors_within_limits(lower_limit=0.899999, upper_limit=0.900001)

        assert vector == pytest.approx([0.9, 1.8], 1.0e-4)

        vector = mapper.random_vector_from_priors_within_limits(lower_limit=0.2, upper_limit=0.3)

        assert vector == pytest.approx([0.21467, 0.4184677], 1.0e-4)

    def test_random_vector_from_prior(self):
        mapper = af.ModelMapper()
        mapper.mock_class = af.PriorModel(mock.MockClassx2)

        np.random.seed(1)

        assert mapper.random_vector_from_priors == pytest.approx(
            [0.41702, 1.44064], 1.0e-4
        )
        assert mapper.random_vector_from_priors == pytest.approx(
            [0.00011437, 0.6046651], 1.0e-4
        )

        # By default, this seeded random will draw a value < -0.15, which is below the lower limit below. This
        # test ensures that this value is resampled to the next draw, which is above 0.15

        mapper.mock_class.one.lower_limit = 0.15

        assert mapper.random_vector_from_priors == pytest.approx(
            [0.27474, 0.1846771], 1.0e-4
        )

    def test_vector_from_prior_medians(self):
        mapper = af.ModelMapper()
        mapper.mock_class = af.PriorModel(mock.MockClassx2)

        assert mapper.physical_values_from_prior_medians == [0.5, 1.0]


class TestUtility:
    def test_prior_prior_model_dict(self):
        mapper = af.ModelMapper(mock_class=mock.MockClassx2)

        assert len(mapper.prior_prior_model_dict) == 2
        assert (
                mapper.prior_prior_model_dict[mapper.prior_tuples_ordered_by_id[0][1]].cls
                == mock.MockClassx2
        )
        assert (
                mapper.prior_prior_model_dict[mapper.prior_tuples_ordered_by_id[1][1]].cls
                == mock.MockClassx2
        )

    def test_name_for_prior(self):
        mapper = af.ModelMapper(mock_class=mock.MockClassx2)

        assert mapper.name_for_prior(mapper.priors[0]) == "mock_class_one"
        assert mapper.name_for_prior(mapper.priors[1]) == "mock_class_two"


class TestPriorReplacement:
    def test_prior_replacement(self):
        mapper = af.ModelMapper(mock_class=mock.MockClassx2)
        result = mapper.mapper_from_gaussian_tuples([(10, 3), (5, 3)])

        assert isinstance(result.mock_class.one, af.GaussianPrior)

    def test_replace_priors_with_gaussians_from_tuples(self):
        mapper = af.ModelMapper(mock_class=mock.MockClassx2)
        result = mapper.mapper_from_gaussian_tuples([(10, 3), (5, 3)])

        assert isinstance(result.mock_class.one, af.GaussianPrior)

    def test_replacing_priors_for_profile(self):
        mapper = af.ModelMapper(mock_class=mock.MockClassx3TupleFloat)
        result = mapper.mapper_from_gaussian_tuples([(10, 3), (5, 3), (5, 3)])

        assert isinstance(
            result.mock_class.one_tuple.unique_prior_tuples[0][1], af.GaussianPrior
        )
        assert isinstance(
            result.mock_class.one_tuple.unique_prior_tuples[1][1], af.GaussianPrior
        )
        assert isinstance(result.mock_class.two, af.GaussianPrior)

    def test_replace_priors_for_two_classes(self):
        mapper = af.ModelMapper(one=mock.MockClassx2, two=mock.MockClassx2)

        result = mapper.mapper_from_gaussian_tuples([(1, 1), (2, 1), (3, 1), (4, 1)])

        assert result.one.one.mean == 1
        assert result.one.two.mean == 2
        assert result.two.one.mean == 3
        assert result.two.two.mean == 4


class TestArguments:
    def test_same_argument_name(self):
        mapper = af.ModelMapper()

        mapper.one = af.PriorModel(mock.MockClassx2)
        mapper.two = af.PriorModel(mock.MockClassx2)

        instance = mapper.instance_from_vector([0.1, 0.2, 0.3, 0.4])

        assert instance.one.one == 0.1
        assert instance.one.two == 0.2
        assert instance.two.one == 0.3
        assert instance.two.two == 0.4


class TestIndependentPriorModel:
    def test_associate_prior_model(self):
        prior_model = af.PriorModel(mock.MockClassx2)

        mapper = af.ModelMapper()

        mapper.prior_model = prior_model

        assert len(mapper.prior_model_tuples) == 1

        instance = mapper.instance_from_vector([0.1, 0.2])

        assert instance.prior_model.one == 0.1
        assert instance.prior_model.two == 0.2


@pytest.fixture(name="list_prior_model")
def make_list_prior_model():
    return af.CollectionPriorModel(
        [af.PriorModel(mock.MockClassx2), af.PriorModel(mock.MockClassx2)]
    )


class TestListPriorModel:
    def test_instance_from_vector(self, list_prior_model):
        mapper = af.ModelMapper()
        mapper.list = list_prior_model

        instance = mapper.instance_from_vector([0.1, 0.2, 0.3, 0.4])

        assert isinstance(instance.list, af.ModelInstance)
        print(instance.list.items)
        assert len(instance.list) == 2
        assert instance.list[0].one == 0.1
        assert instance.list[0].two == 0.2
        assert instance.list[1].one == 0.3
        assert instance.list[1].two == 0.4

    def test_prior_results_for_gaussian_tuples(self, list_prior_model):
        mapper = af.ModelMapper()
        mapper.list = list_prior_model

        gaussian_mapper = mapper.mapper_from_gaussian_tuples(
            [(1, 5), (2, 5), (3, 5), (4, 5)]
        )

        assert len(gaussian_mapper.list) == 2
        assert gaussian_mapper.list[0].one.mean == 1
        assert gaussian_mapper.list[0].two.mean == 2
        assert gaussian_mapper.list[1].one.mean == 3
        assert gaussian_mapper.list[1].two.mean == 4
        assert gaussian_mapper.list[0].one.sigma == 5
        assert gaussian_mapper.list[0].two.sigma == 5
        assert gaussian_mapper.list[1].one.sigma == 5
        assert gaussian_mapper.list[1].two.sigma == 5

    def test_prior_results_for_gaussian_tuples__include_override_from_width_file(
            self, list_prior_model
    ):
        mapper = af.ModelMapper()
        mapper.list = list_prior_model

        gaussian_mapper = mapper.mapper_from_gaussian_tuples(
            [(1, 0), (2, 0), (3, 0), (4, 0)]
        )

        assert len(gaussian_mapper.list) == 2
        assert gaussian_mapper.list[0].one.mean == 1
        assert gaussian_mapper.list[0].two.mean == 2
        assert gaussian_mapper.list[1].one.mean == 3
        assert gaussian_mapper.list[1].two.mean == 4
        assert gaussian_mapper.list[0].one.sigma == 1
        assert gaussian_mapper.list[0].two.sigma == 2
        assert gaussian_mapper.list[1].one.sigma == 1
        assert gaussian_mapper.list[1].two.sigma == 2

    def test_automatic_boxing(self):
        mapper = af.ModelMapper()
        mapper.list = [af.PriorModel(mock.MockClassx2), af.PriorModel(mock.MockClassx2)]

        assert isinstance(mapper.list, af.CollectionPriorModel)


@pytest.fixture(name="mock_with_instance")
def make_mock_with_instance():
    mock_with_instance = af.PriorModel(mock.MockClassx2)
    mock_with_instance.one = 3.0
    return mock_with_instance


class Testinstance:
    def test_instance_prior_count(self, mock_with_instance):
        mapper = af.ModelMapper()
        mapper.mock_class = mock_with_instance

        assert len(mapper.unique_prior_tuples) == 1

    def test_retrieve_instances(self, mock_with_instance):
        assert len(mock_with_instance.instance_tuples) == 1

    def test_instance_prior_reconstruction(self, mock_with_instance):
        mapper = af.ModelMapper()
        mapper.mock_class = mock_with_instance

        instance = mapper.instance_for_arguments({mock_with_instance.two: 0.5})

        assert instance.mock_class.one == 3
        assert instance.mock_class.two == 0.5

    def test_instance_in_config(self):
        mapper = af.ModelMapper()

        mock_with_instance = af.PriorModel(MockClassx2instance, one=3)

        mapper.mock_class = mock_with_instance

        instance = mapper.instance_for_arguments({mock_with_instance.two: 0.5})

        assert instance.mock_class.one == 3
        assert instance.mock_class.two == 0.5

    def test_set_float(self):
        prior_model = af.PriorModel(mock.MockClassx2)
        prior_model.one = 3
        prior_model.two = 4.0
        assert prior_model.one == 3
        assert prior_model.two == 4.0

    def test_list_prior_model_instances(self, mapper):
        prior_model = af.PriorModel(mock.MockClassx2)
        prior_model.one = 3.0
        prior_model.two = 4.0

        mapper.mock_list = [prior_model]
        assert isinstance(mapper.mock_list, af.CollectionPriorModel)
        assert len(mapper.instance_tuples) == 2

    def test_set_for_tuple_prior(self):
        prior_model = af.PriorModel(mock_real.EllipticalSersic)
        prior_model.centre_0 = 1.0
        prior_model.centre_1 = 2.0
        prior_model.axis_ratio = 1.0
        prior_model.phi = 1.0
        prior_model.intensity = 1.0
        prior_model.effective_radius = 1.0
        prior_model.sersic_index = 1.0
        instance = prior_model.instance_for_arguments({})
        assert instance.centre == (1.0, 2.0)


@pytest.fixture(name="mock_config")
def make_mock_config():
    return


@pytest.fixture(name="mapper")
def make_mapper():
    return af.ModelMapper()


@pytest.fixture(name="mapper_with_one")
def make_mapper_with_one():
    mapper = af.ModelMapper()
    mapper.one = af.PriorModel(mock.MockClassx2)
    return mapper


@pytest.fixture(name="mapper_with_list")
def make_mapper_with_list():
    mapper = af.ModelMapper()
    mapper.list = [af.PriorModel(mock.MockClassx2), af.PriorModel(mock.MockClassx2)]
    return mapper


class TestGaussianWidthConfig:
    def test_relative_widths(self, mapper):
        mapper.relative_width = mock.MockClassRelativeWidth
        new_mapper = mapper.mapper_from_gaussian_tuples([(1, 0), (1, 0), (1, 0)])

        assert new_mapper.relative_width.one.mean == 1.0
        assert new_mapper.relative_width.one.sigma == 0.1

        assert new_mapper.relative_width.two.mean == 1.0
        assert new_mapper.relative_width.two.sigma == 0.5

        assert new_mapper.relative_width.three.mean == 1.0
        assert new_mapper.relative_width.three.sigma == 1.0

    def test_prior_classes(self, mapper_with_one):
        assert mapper_with_one.prior_class_dict == {
            mapper_with_one.one.one: mock.MockClassx2,
            mapper_with_one.one.two: mock.MockClassx2,
        }

    def test_prior_classes_list(self, mapper_with_list):
        assert mapper_with_list.prior_class_dict == {
            mapper_with_list.list[0].one: mock.MockClassx2,
            mapper_with_list.list[0].two: mock.MockClassx2,
            mapper_with_list.list[1].one: mock.MockClassx2,
            mapper_with_list.list[1].two: mock.MockClassx2,
        }

    def test_no_override(self):
        mapper = af.ModelMapper()

        mapper.one = af.PriorModel(mock.MockClassx2)

        af.ModelMapper()

        assert mapper.one is not None


@pytest.fixture(name="promise_mapper")
def make_promise_mapper():
    mapper = af.ModelMapper()
    mapper.component = af.PriorModel(
        mock.MockComponents,
        parameter=af.prior.Promise(
            None,
            None,
            is_instance=False,
            result_path=None,
            assert_exists=False
        ),
    )
    return mapper


class TestPromises:
    def test_promise_count(self, promise_mapper):
        assert promise_mapper.promise_count == 1
        assert promise_mapper.variable_promise_count == 1

        promise_mapper.component.parameter.is_instance = True
        assert promise_mapper.variable_promise_count == 0

    def test_raises(self, promise_mapper):
        with pytest.raises(exc.PriorException):
            promise_mapper.instance_from_prior_medians()
