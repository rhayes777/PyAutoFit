import os
import pytest

import autofit as af
from autoconf import conf

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(name="mapper")
def make_mapper():
    return af.ModelMapper()


@pytest.fixture(name="mock_list")
def make_mock_list():
    return [af.Model(af.m.MockClassx4), af.Model(af.m.MockClassx4)]


@pytest.fixture(name="result")
def make_result():
    mapper = af.ModelMapper()
    mapper.component = af.m.MockClassx2Tuple
    # noinspection PyTypeChecker
    return af.mock.MockResult(
        samples_summary=af.m.MockSamplesSummary(
            model=mapper,
            prior_means=[0, 1],
        ),
    )


def test__environment_variable_override():
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
    os.environ["NUMEXPR_NUM_THREADS"] = "2"

    conf.instance["general"]["parallel"]["warn_environment_variables"] = True

    with pytest.warns(af.exc.SearchWarning):
        af.mock.MockSearch(number_of_cores=2)

    conf.instance["general"]["parallel"]["warn_environment_variables"] = False

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
            result.model.mapper_from_prior_means(
                result.samples_summary.prior_means, a=2.0, r=1.0
            )


class TestLabels:
    def test_param_names(self):
        model = af.Model(af.m.MockClassx4)
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
