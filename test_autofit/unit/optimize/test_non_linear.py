import os
import shutil

import pytest
from autoconf import conf
import autofit as af
from autofit import Paths
from test_autofit.mock import (
    GeometryProfile,
    MockClassNLOx4,
)
from autofit.optimize.non_linear.mock_nlo import MockSamples

directory = os.path.dirname(os.path.realpath(__file__))
pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        config_path=os.path.join(directory, "files/nlo/config"),
        output_path=os.path.join(directory, "files/nlo/output")
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
    return af.Result(
        samples=MockSamples(gaussian_tuples=[(0, 0), (1, 0)]),
        previous_model=mapper,
    )


class TestResult:
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


class TestCopyWithNameExtension:
    @staticmethod
    def assert_non_linear_attributes_equal(copy):
        assert copy.paths.phase_name == "phase_name/one"

    def test_copy_with_name_extension(self):
        optimizer = af.MockNLO(Paths("phase_name", phase_tag="tag"))
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
    nlo_setup_path = "{}/files/nlo/setup/".format(
        os.path.dirname(os.path.realpath(__file__))
    )

    if os.path.exists(nlo_setup_path):
        shutil.rmtree(nlo_setup_path)

    os.mkdir(nlo_setup_path)

    return nlo_setup_path


@pytest.fixture(name="nlo_model_info_path")
def test_nlo_model_info():
    nlo_model_info_path = "{}/files/nlo/model_info/".format(
        os.path.dirname(os.path.realpath(__file__))
    )

    if os.path.exists(nlo_model_info_path):
        shutil.rmtree(nlo_model_info_path)

    return nlo_model_info_path


@pytest.fixture(name="nlo_wrong_info_path")
def test_nlo_wrong_info():
    nlo_wrong_info_path = "{}/files/nlo/wrong_info/".format(
        os.path.dirname(os.path.realpath(__file__))
    )

    if os.path.exists(nlo_wrong_info_path):
        shutil.rmtree(nlo_wrong_info_path)

    os.mkdir(nlo_wrong_info_path)

    return nlo_wrong_info_path


class TestDirectorySetup:
    def test__1_class__correct_directory(self, nlo_setup_path):
        af.conf.instance.output_path = nlo_setup_path + "1_class"
        af.MockNLO(Paths(phase_name=""))

        assert os.path.exists(nlo_setup_path + "1_class")



class TestLabels:
    def test_param_names(self):
        model = af.PriorModel(MockClassNLOx4)
        assert [
                   "one",
                   "two",
                   "three",
                   "four",
               ] == model.param_names

    def test_label_config(self):
        assert af.conf.instance.label.label("one") == "x4p0"
        assert af.conf.instance.label.label("two") == "x4p1"
        assert af.conf.instance.label.label("three") == "x4p2"
        assert af.conf.instance.label.label("four") == "x4p3"