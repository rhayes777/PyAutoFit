import os
from os import path
import pickle
import shutil

import numpy as np
import pytest

import autofit as af
from autoconf import conf
from autofit.mock import mock
from autofit.mock.mock_search import MockSamples

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(name="mapper")
def make_mapper():
    return af.ModelMapper()


@pytest.fixture(name="mock_list")
def make_mock_list():
    return [af.PriorModel(mock.MockClassx4), af.PriorModel(mock.MockClassx4)]


@pytest.fixture(name="result")
def make_result():
    mapper = af.ModelMapper()
    mapper.component = mock.MockClassx2Tuple
    # noinspection PyTypeChecker
    return af.Result(
        samples=MockSamples(gaussian_tuples=[(0, 0), (1, 0)]),
        previous_model=mapper,
        search=mock.MockSearch(),
    )


class TestResult:
    def test_model(self, result):
        component = result.model.component
        assert component.one_tuple.one_tuple_0.mean == 0
        assert component.one_tuple.one_tuple_1.mean == 1
        assert component.one_tuple.one_tuple_0.sigma == 0.2
        assert component.one_tuple.one_tuple_1.sigma == 0.2

    def test_model_absolute(self, result):
        component = result.model_absolute(a=2.0).component
        assert component.one_tuple.one_tuple_0.mean == 0
        assert component.one_tuple.one_tuple_1.mean == 1
        assert component.one_tuple.one_tuple_0.sigma == 2.0
        assert component.one_tuple.one_tuple_1.sigma == 2.0

    def test_model_relative(self, result):
        component = result.model_relative(r=1.0).component
        assert component.one_tuple.one_tuple_0.mean == 0
        assert component.one_tuple.one_tuple_1.mean == 1
        assert component.one_tuple.one_tuple_0.sigma == 0.0
        assert component.one_tuple.one_tuple_1.sigma == 1.0

    def test_raises(self, result):
        with pytest.raises(af.exc.PriorException):
            result.model.mapper_from_gaussian_tuples(
                result.samples.gaussian_tuples, a=2.0, r=1.0
            )


class TestCopyWithNameExtension:
    @staticmethod
    def assert_non_linear_attributes_equal(copy):
        assert copy.paths.name ==  path.join("name", "one")

    def test_copy_with_name_extension(self):
        search = af.MockSearch(af.Paths("name", tag="tag"))
        copy = search.copy_with_name_extension("one")

        self.assert_non_linear_attributes_equal(copy)
        assert search.paths.tag == copy.paths.tag



class TestLabels:
    def test_param_names(self):
        model = af.PriorModel(mock.MockClassx4)
        assert [
                   "one",
                   "two",
                   "three",
                   "four",
               ] == model.model_component_and_parameter_names

    def test_label_config(self):
        assert conf.instance["notation"]["label"]["label"]["one"] == "one_label"
        assert conf.instance["notation"]["label"]["label"]["two"] == "two_label"
        assert conf.instance["notation"]["label"]["label"]["three"] == "three_label"
        assert conf.instance["notation"]["label"]["label"]["four"] == "four_label"


test_path = path.join(
    "{}".format(path.dirname(path.realpath(__file__))), "files", "phase"
)


class TestMovePickleFiles:
    def test__move_pickle_files(self):

        search = af.MockSearch(paths=af.Paths(name="pickles", path_prefix=path.join("non_linear", "abstract_search")))

        pickle_paths = [
            path.join(
                conf.instance.output_path, "non_linear", "abstract_search", "pickles"
            )
        ]

        arr = np.ones((3, 3))

        with open(path.join(pickle_paths[0], "test.pickle"), "wb") as f:
            pickle.dump(arr, f)

        pickle_paths = [
            path.join(
                conf.instance.output_path,
                "non_linear",
                "abstract_search",
                "pickles",
                "test.pickle",
            )
        ]

        search.move_pickle_files(pickle_files=pickle_paths)

        with open(path.join(pickle_paths[0]), "rb") as f:
            arr_load = pickle.load(f)

        assert (arr == arr_load).all()

        if path.exists(test_path):
            shutil.rmtree(test_path)
