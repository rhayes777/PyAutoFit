import itertools
import os
import shutil

import pytest

import autofit as af
from autofit import ModelMapper, Paths
from autofit.optimize.non_linear.output import AbstractOutput
from test_autofit.mock import (
    GeometryProfile,
    MockClassNLOx4,
    MockClassNLOx5,
    MockNonLinearOptimizer,
    MockClassNLOx6,
)

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(scope="session", autouse=True)
def do_something():
    af.conf.instance = af.conf.Config(
        "{}/../test_files/configs/non_linear".format(
            os.path.dirname(os.path.realpath(__file__))
        )
    )


@pytest.fixture(name="mapper")
def make_mapper():
    return af.ModelMapper()


@pytest.fixture(name="mock_list")
def make_mock_list():
    return [af.PriorModel(MockClassNLOx4), af.PriorModel(MockClassNLOx4)]


@pytest.fixture(name="result")
def make_result():
    mapper = af.ModelMapper()
    mapper.profile = GeometryProfile
    # noinspection PyTypeChecker
    return af.Result(None, None, mapper, [(0, 0), (1, 0)])


class TestResult(object):
    def test_model(self, result):
        profile = result.model.profile
        assert profile.centre_0.mean == 0
        assert profile.centre_1.mean == 1
        assert profile.centre_0.sigma == 0.05
        assert profile.centre_1.sigma == 0.05

    def test_model_absolute(self, result):
        profile = result.model_absolute(a=2.0).profile
        assert profile.centre_0.mean == 0
        assert profile.centre_1.mean == 1
        assert profile.centre_0.sigma == 2.0
        assert profile.centre_1.sigma == 2.0

    def test_model_relative(self, result):
        profile = result.model_relative(r=1.0).profile
        assert profile.centre_0.mean == 0
        assert profile.centre_1.mean == 1
        assert profile.centre_0.sigma == 0.0
        assert profile.centre_1.sigma == 1.0

    def test_raises(self, result):
        with pytest.raises(af.exc.PriorException):
            result.model.mapper_from_gaussian_tuples(
                result.gaussian_tuples, a=2.0, r=1.0
            )


class TestCopyWithNameExtension(object):
    @staticmethod
    def assert_non_linear_attributes_equal(copy):
        assert copy.paths.phase_name == "phase_name/one"

    def test_copy_with_name_extension(self):
        optimizer = af.NonLinearOptimizer(Paths("phase_name", phase_tag="tag"))
        copy = optimizer.copy_with_name_extension("one")

        self.assert_non_linear_attributes_equal(copy)
        assert optimizer.paths.phase_tag == copy.paths.phase_tag

    def test_grid_search(self):
        optimizer = af.GridSearch(Paths("phase_name"), step_size=17, grid=lambda x: x)

        copy = optimizer.copy_with_name_extension("one")
        self.assert_non_linear_attributes_equal(copy)
        assert isinstance(copy, af.GridSearch)
        assert copy.step_size is optimizer.step_size
        assert copy.grid is optimizer.grid


@pytest.fixture(name="nlo_setup_path")
def test_nlo_setup():
    nlo_setup_path = "{}/../test_files/non_linear/nlo/setup/".format(
        os.path.dirname(os.path.realpath(__file__))
    )

    if os.path.exists(nlo_setup_path):
        shutil.rmtree(nlo_setup_path)

    os.mkdir(nlo_setup_path)

    return nlo_setup_path


@pytest.fixture(name="nlo_model_info_path")
def test_nlo_model_info():
    nlo_model_info_path = "{}/../test_files/non_linear/nlo/model_info/".format(
        os.path.dirname(os.path.realpath(__file__))
    )

    if os.path.exists(nlo_model_info_path):
        shutil.rmtree(nlo_model_info_path)

    return nlo_model_info_path


@pytest.fixture(name="nlo_wrong_info_path")
def test_nlo_wrong_info():
    nlo_wrong_info_path = "{}/../test_files/non_linear/nlo/wrong_info/".format(
        os.path.dirname(os.path.realpath(__file__))
    )

    if os.path.exists(nlo_wrong_info_path):
        shutil.rmtree(nlo_wrong_info_path)

    os.mkdir(nlo_wrong_info_path)

    return nlo_wrong_info_path


class TestDirectorySetup:
    def test__1_class__correct_directory(self, nlo_setup_path):
        af.conf.instance.output_path = nlo_setup_path + "1_class"
        af.NonLinearOptimizer(Paths(phase_name=""))

        assert os.path.exists(nlo_setup_path + "1_class")


class TestMostProbableAndLikely(object):
    def test__most_probable_parameters_and_instance__2_classes_6_params(self):
        mapper = af.ModelMapper(
            mock_class_1=MockClassNLOx4, mock_class_2=MockClassNLOx6
        )
        nlo = MockNonLinearOptimizer(
            phase_name="",
            model_mapper=mapper,
            most_probable=[1.0, 2.0, 3.0, 4.0, -5.0, -6.0, -7.0, -8.0, 9.0, 10.0],
        )

        most_probable = nlo.most_probable_model_instance

        assert most_probable.mock_class_1.one == 1.0
        assert most_probable.mock_class_1.two == 2.0
        assert most_probable.mock_class_1.three == 3.0
        assert most_probable.mock_class_1.four == 4.0

        assert most_probable.mock_class_2.one == (-5.0, -6.0)
        assert most_probable.mock_class_2.two == (-7.0, -8.0)
        assert most_probable.mock_class_2.three == 9.0
        assert most_probable.mock_class_2.four == 10.0

    def test__most_probable__setup_model_instance__1_class_5_params_but_1_is_instance(
        self
    ):
        mapper = af.ModelMapper(mock_class=MockClassNLOx5)
        mapper.mock_class.five = 10.0

        nlo = MockNonLinearOptimizer(
            phase_name="",
            model_mapper=mapper,
            most_probable=[1.0, -2.0, 3.0, 4.0, 10.0],
        )

        most_probable = nlo.most_probable_model_instance

        assert most_probable.mock_class.one == 1.0
        assert most_probable.mock_class.two == -2.0
        assert most_probable.mock_class.three == 3.0
        assert most_probable.mock_class.four == 4.0
        assert most_probable.mock_class.five == 10.0

    def test__most_likely_parameters_and_instance__2_classes_6_params(self):
        mapper = af.ModelMapper(
            mock_class_1=MockClassNLOx4, mock_class_2=MockClassNLOx6
        )
        nlo = MockNonLinearOptimizer(
            phase_name="",
            model_mapper=mapper,
            most_likely=[21.0, 22.0, 23.0, 24.0, 25.0, -26.0, -27.0, 28.0, 29.0, 30.0],
        )

        most_likely = nlo.most_likely_model_instance

        assert most_likely.mock_class_1.one == 21.0
        assert most_likely.mock_class_1.two == 22.0
        assert most_likely.mock_class_1.three == 23.0
        assert most_likely.mock_class_1.four == 24.0

        assert most_likely.mock_class_2.one == (25.0, -26.0)
        assert most_likely.mock_class_2.two == (-27.0, 28.0)
        assert most_likely.mock_class_2.three == 29.0
        assert most_likely.mock_class_2.four == 30.0

    def test__most_likely__setup_model_instance__1_class_5_params_but_1_is_instance(
        self
    ):
        mapper = af.ModelMapper(mock_class=MockClassNLOx5)
        mapper.mock_class.five = 10.0
        nlo = MockNonLinearOptimizer(
            phase_name="",
            model_mapper=mapper,
            most_likely=[9.0, -10.0, -11.0, 12.0, 10.0],
        )

        most_likely = nlo.most_likely_model_instance

        assert most_likely.mock_class.one == 9.0
        assert most_likely.mock_class.two == -10.0
        assert most_likely.mock_class.three == -11.0
        assert most_likely.mock_class.four == 12.0
        assert most_likely.mock_class.five == 10.0


class TestGaussianPriors(object):
    def test__1_class__gaussian_priors_at_3_sigma_confidence(self):
        mapper = af.ModelMapper(mock_class=MockClassNLOx4)
        nlo = MockNonLinearOptimizer(
            phase_name="",
            model_mapper=mapper,
            most_probable=[1.0, 2.0, 3.0, 4.1],
            model_lower_params=[0.88, 1.88, 2.88, 3.88],
            model_upper_params=[1.12, 2.12, 3.12, 4.12],
        )

        gaussian_priors = nlo.gaussian_priors_at_sigma_limit(sigma_limit=3.0)

        assert gaussian_priors[0][0] == 1.0
        assert gaussian_priors[1][0] == 2.0
        assert gaussian_priors[2][0] == 3.0
        assert gaussian_priors[3][0] == 4.1

        assert gaussian_priors[0][1] == pytest.approx(0.12, 1e-2)
        assert gaussian_priors[1][1] == pytest.approx(0.12, 1e-2)
        assert gaussian_priors[2][1] == pytest.approx(0.12, 1e-2)
        assert gaussian_priors[3][1] == pytest.approx(0.22, 1e-2)


class TestOffsetFromInput:
    def test__input_model_offset_from_most_probable__parameters_and_instance__1_class_4_params(
        self
    ):
        mapper = af.ModelMapper(mock_class=MockClassNLOx4)
        nlo = MockNonLinearOptimizer(
            phase_name="", model_mapper=mapper, most_probable=[1.0, -2.0, 3.0, 4.0]
        )

        offset_values = nlo.values_offset_from_input_model_parameters(
            input_model_parameters=[1.0, 1.0, 2.0, 3.0]
        )

        assert offset_values == [0.0, -3.0, 1.0, 1.0]

        mapper = af.ModelMapper(
            mock_class_1=MockClassNLOx4, mock_class_2=MockClassNLOx6
        )
        nlo = MockNonLinearOptimizer(
            phase_name="",
            model_mapper=mapper,
            most_probable=[1.0, 2.0, 3.0, 4.0, -5.0, -6.0, -7.0, -8.0, 9.0, 10.0],
        )

        offset_values = nlo.values_offset_from_input_model_parameters(
            input_model_parameters=[
                1.0,
                1.0,
                2.0,
                3.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                20.0,
            ]
        )

        assert offset_values == [
            0.0,
            1.0,
            1.0,
            1.0,
            -15.0,
            -16.0,
            -17.0,
            -18.0,
            -1.0,
            -10.0,
        ]


@pytest.fixture(name="optimizer")
def make_optimizer():
    return AbstractOutput(ModelMapper(), Paths(phase_name=""))


class TestLabels(object):
    def test_param_names(self, optimizer):
        optimizer.model.prior_model = MockClassNLOx4
        assert [
            "prior_model_one",
            "prior_model_two",
            "prior_model_three",
            "prior_model_four",
        ] == optimizer.model.param_names

    def test_properties(self, optimizer):
        optimizer.model.prior_model = MockClassNLOx4

        assert len(optimizer.param_labels) == 4
        assert len(optimizer.model.param_names) == 4

    def test_label_config(self):
        assert af.conf.instance.label.label("one") == "x4p0"
        assert af.conf.instance.label.label("two") == "x4p1"
        assert af.conf.instance.label.label("three") == "x4p2"
        assert af.conf.instance.label.label("four") == "x4p3"

    def test_labels(self, optimizer):
        af.AbstractPriorModel._ids = itertools.count()
        optimizer.model.prior_model = MockClassNLOx4

        assert optimizer.param_labels == [
            r"x4p0_{\mathrm{a1}}",
            r"x4p1_{\mathrm{a1}}",
            r"x4p2_{\mathrm{a1}}",
            r"x4p3_{\mathrm{a1}}",
        ]
