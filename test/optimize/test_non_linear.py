import itertools
import os
import shutil
from functools import wraps

import pytest

import autofit.mapper.prior_model
from autofit.mapper import model_mapper, prior as p
import autofit.optimize.non_linear.downhill_simplex
import autofit.optimize.non_linear.grid_search
import autofit.optimize.non_linear.multi_nest
import autofit.optimize.non_linear.non_linear
from autofit import conf
from autofit import exc
from autofit import mock
from autofit.mapper import model_mapper
from test.mock.mock import MockClassNLOx4, MockClassNLOx5, MockClassNLOx6, MockAnalysis, MockNonLinearOptimizer

pytestmark = pytest.mark.filterwarnings('ignore::FutureWarning')

@pytest.fixture(scope="session", autouse=True)
def do_something():
    conf.instance = conf.Config(
        "{}/../test_files/configs/non_linear".format(os.path.dirname(os.path.realpath(__file__))))


@pytest.fixture(name='mapper')
def make_mapper():
    return model_mapper.ModelMapper()


@pytest.fixture(name="mock_list")
def make_mock_list():
    return [autofit.mapper.prior_model.PriorModel(MockClassNLOx4),
            autofit.mapper.prior_model.PriorModel(MockClassNLOx4)]


@pytest.fixture(name="result")
def make_result():
    mapper = model_mapper.ModelMapper()
    mapper.profile = mock.GeometryProfile
    # noinspection PyTypeChecker
    return autofit.optimize.non_linear.non_linear.Result(None, None, mapper, [(0, 0), (1, 0)])


class TestResult(object):

    def test_variable(self, result):
        profile = result.variable.profile
        assert profile.centre_0.mean == 0
        assert profile.centre_1.mean == 1
        assert profile.centre_0.sigma == 0.05
        assert profile.centre_1.sigma == 0.05

    def test_variable_absolute(self, result):
        profile = result.variable_absolute(a=2.0).profile
        assert profile.centre_0.mean == 0
        assert profile.centre_1.mean == 1
        assert profile.centre_0.sigma == 2.0
        assert profile.centre_1.sigma == 2.0

    def test_variable_relative(self, result):
        profile = result.variable_relative(r=1.0).profile
        assert profile.centre_0.mean == 0
        assert profile.centre_1.mean == 1
        assert profile.centre_0.sigma == 0.0
        assert profile.centre_1.sigma == 1.0

    def test_raises(self, result):
        with pytest.raises(exc.PriorException):
            result.variable.mapper_from_gaussian_tuples(result.gaussian_tuples, a=2.0, r=1.0)


class TestCopyWithNameExtension(object):

    @staticmethod
    def assert_non_linear_attributes_equal(copy, optimizer):
        assert copy.phase_name == "phase_name/one"
        assert copy.variable == optimizer.variable

    def test_copy_with_name_extension(self):
        optimizer = autofit.optimize.non_linear.non_linear.NonLinearOptimizer("phase_name")
        copy = optimizer.copy_with_name_extension("one")

        self.assert_non_linear_attributes_equal(copy, optimizer)

    def test_grid_search(self):
        optimizer = autofit.optimize.non_linear.grid_search.GridSearch("phase_name", step_size=17, grid=lambda x: x)

        copy = optimizer.copy_with_name_extension("one")
        self.assert_non_linear_attributes_equal(copy, optimizer)
        assert isinstance(copy, autofit.optimize.non_linear.grid_search.GridSearch)
        assert copy.step_size is optimizer.step_size
        assert copy.grid is optimizer.grid


# noinspection PyUnresolvedReferences
class TestParamNames(object):

    def test_label_prior_model_tuples(self, mapper, mock_list):
        mapper.mock_list = mock_list

        assert [tup.name for tup in mapper.mock_list.label_prior_model_tuples] == ['0', '1']

    def test_label_prior_model_tuples_with_mapping_name(self, mapper):
        one = autofit.mapper.prior_model.PriorModel(MockClassNLOx4)
        two = autofit.mapper.prior_model.PriorModel(MockClassNLOx4)

        one.mapping_name = "one"
        two.mapping_name = "two"

        mapper.mock_list = [one, two]

        assert [tup.name for tup in mapper.mock_list.label_prior_model_tuples] == ['one', 'two']

    def test_prior_prior_model_name_dict(self, mapper, mock_list):
        mapper.mock_list = mock_list
        prior_prior_model_name_dict = mapper.prior_prior_model_name_dict

        assert len({value for key, value in prior_prior_model_name_dict.items()}) == 2


@pytest.fixture(name='nlo_setup_path')
def test_nlo_setup():
    nlo_setup_path = "{}/../test_files/non_linear/nlo/setup/".format(os.path.dirname(os.path.realpath(__file__)))

    if os.path.exists(nlo_setup_path):
        shutil.rmtree(nlo_setup_path)

    os.mkdir(nlo_setup_path)

    return nlo_setup_path


@pytest.fixture(name='nlo_paramnames_path')
def test_nlo_paramnames():
    nlo_paramnames_path = "{}/../test_files/non_linear/nlo/paramnames/".format(
        os.path.dirname(os.path.realpath(__file__)))

    if os.path.exists(nlo_paramnames_path):
        shutil.rmtree(nlo_paramnames_path)

    return nlo_paramnames_path


@pytest.fixture(name='nlo_model_info_path')
def test_nlo_model_info():
    nlo_model_info_path = "{}/../test_files/non_linear/nlo/model_info/".format(
        os.path.dirname(os.path.realpath(__file__)))

    if os.path.exists(nlo_model_info_path):
        shutil.rmtree(nlo_model_info_path)

    return nlo_model_info_path


@pytest.fixture(name='nlo_wrong_info_path')
def test_nlo_wrong_info():
    nlo_wrong_info_path = "{}/../test_files/non_linear/nlo/wrong_info/".format(
        os.path.dirname(os.path.realpath(__file__)))

    if os.path.exists(nlo_wrong_info_path):
        shutil.rmtree(nlo_wrong_info_path)

    os.mkdir(nlo_wrong_info_path)

    return nlo_wrong_info_path


class TestDirectorySetup:

    def test__1_class__correct_directory(self, nlo_setup_path):

        conf.instance.output_path = nlo_setup_path + '1_class'
        mapper = model_mapper.ModelMapper(mock_class=MockClassNLOx4)
        autofit.optimize.non_linear.non_linear.NonLinearOptimizer(phase_name='', model_mapper=mapper)

        assert os.path.exists(nlo_setup_path + '1_class')


class TestTotalParameters:

    def test__1_class__four_parameters(self, nlo_setup_path):

        conf.instance.output_path = nlo_setup_path + '1_class'
        mapper = model_mapper.ModelMapper(mock_class=MockClassNLOx4)
        nlo = autofit.optimize.non_linear.non_linear.NonLinearOptimizer(phase_name='', model_mapper=mapper)

        assert nlo.variable.prior_count == 4

    def test__2_classes__six_parameters(self, nlo_setup_path):

        conf.instance.output_path = nlo_setup_path + '2_classes'
        mapper = model_mapper.ModelMapper(class_1=MockClassNLOx4, class_2=MockClassNLOx6)
        nlo = autofit.optimize.non_linear.non_linear.NonLinearOptimizer(phase_name='', model_mapper=mapper)

        assert nlo.variable.prior_count == 10

@pytest.fixture(name='nlo_priors_path')
def test_nlo_priors():
    nlo_priors_path = "{}/../test_files/non_linear/nlo/priors".format(os.path.dirname(os.path.realpath(__file__)))

    if os.path.exists(nlo_priors_path):
        shutil.rmtree(nlo_priors_path)

    os.mkdir(nlo_priors_path)

    return nlo_priors_path


def create_path(func):
    @wraps(func)
    def wrapper(path):
        if not os.path.exists(path):
            os.makedirs(path)
        return func(path)

    return wrapper

@create_path
def create_weighted_samples_4_parameters(path):
    with open(path + '/nlo.txt', 'w+') as weighted_samples:
        weighted_samples.write(
            '    0.020000000000000000E+00    0.999999990000000000E+07    0.110000000000000000E+01    '
            '0.210000000000000000E+01    0.310000000000000000E+01    0.410000000000000000E+01\n'
            '    0.020000000000000000E+00    0.999999990000000000E+07    0.090000000000000000E+01    '
            '0.190000000000000000E+01    0.290000000000000000E+01    0.390000000000000000E+01\n'
            '    0.010000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    '
            '0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01\n'
            '    0.050000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    '
            '0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01\n'
            '    0.100000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    '
            '0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01\n'
            '    0.100000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    '
            '0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01\n'
            '    0.100000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    '
            '0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01\n'
            '    0.100000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    '
            '0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01\n'
            '    0.200000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    '
            '0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01\n'
            '    0.300000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    '
            '0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01')


class TestMostProbableAndLikely(object):

    def test__most_probable_parameters_and_instance__2_classes_6_params(self):

        mapper = model_mapper.ModelMapper(mock_class_1=MockClassNLOx4,
                                          mock_class_2=MockClassNLOx6)
        nlo = MockNonLinearOptimizer(phase_name='', model_mapper=mapper,
                                     most_probable=[1.0, 2.0, 3.0, 4.0, -5.0, -6.0, -7.0, -8.0, 9.0, 10.0])

        most_probable = nlo.most_probable_model_instance

        assert most_probable.mock_class_1.one == 1.0
        assert most_probable.mock_class_1.two == 2.0
        assert most_probable.mock_class_1.three == 3.0
        assert most_probable.mock_class_1.four == 4.0

        assert most_probable.mock_class_2.one == (-5.0, -6.0)
        assert most_probable.mock_class_2.two == (-7.0, -8.0)
        assert most_probable.mock_class_2.three == 9.0
        assert most_probable.mock_class_2.four == 10.0

    def test__most_probable__setup_model_instance__1_class_5_params_but_1_is_constant(self):

        mapper = model_mapper.ModelMapper(mock_class=MockClassNLOx5)
        mapper.mock_class.five = p.Constant(10.0)

        nlo = MockNonLinearOptimizer(phase_name='', model_mapper=mapper, most_probable=[1.0, -2.0, 3.0, 4.0, 10.0])

        most_probable = nlo.most_probable_model_instance

        assert most_probable.mock_class.one == 1.0
        assert most_probable.mock_class.two == -2.0
        assert most_probable.mock_class.three == 3.0
        assert most_probable.mock_class.four == 4.0
        assert most_probable.mock_class.five == 10.0

    def test__most_likely_parameters_and_instance__2_classes_6_params(self):

        mapper = model_mapper.ModelMapper(mock_class_1=MockClassNLOx4,
                                          mock_class_2=MockClassNLOx6)
        nlo = MockNonLinearOptimizer(phase_name='', model_mapper=mapper,
                                     most_likely=[21.0, 22.0, 23.0, 24.0, 25.0, -26.0, -27.0, 28.0, 29.0, 30.0])

        most_likely = nlo.most_likely_model_instance

        assert most_likely.mock_class_1.one == 21.0
        assert most_likely.mock_class_1.two == 22.0
        assert most_likely.mock_class_1.three == 23.0
        assert most_likely.mock_class_1.four == 24.0

        assert most_likely.mock_class_2.one == (25.0, -26.0)
        assert most_likely.mock_class_2.two == (-27.0, 28.0)
        assert most_likely.mock_class_2.three == 29.0
        assert most_likely.mock_class_2.four == 30.0

    def test__most_likely__setup_model_instance__1_class_5_params_but_1_is_constant(self):

        mapper = model_mapper.ModelMapper(mock_class=MockClassNLOx5)
        mapper.mock_class.five = p.Constant(10.0)
        nlo = MockNonLinearOptimizer(phase_name='', model_mapper=mapper, most_likely=[9.0, -10.0, -11.0, 12.0, 10.0])

        most_likely = nlo.most_likely_model_instance

        assert most_likely.mock_class.one == 9.0
        assert most_likely.mock_class.two == -10.0
        assert most_likely.mock_class.three == -11.0
        assert most_likely.mock_class.four == 12.0
        assert most_likely.mock_class.five == 10.0


class TestGaussianPriors(object):

    def test__1_class__gaussian_priors_at_3_sigma_confidence(self, nlo_priors_path):
        
        conf.instance.output_path = nlo_priors_path

        mapper = model_mapper.ModelMapper(mock_class=MockClassNLOx4)
        nlo = MockNonLinearOptimizer(phase_name='', model_mapper=mapper, most_probable=[1.0, 2.0, 3.0, 4.1])

        create_weighted_samples_4_parameters(path=nlo.backup_path)
        gaussian_priors = nlo.gaussian_priors_at_sigma_limit(sigma_limit=3.0)

        assert gaussian_priors[0][0] == 1.0
        assert gaussian_priors[1][0] == 2.0
        assert gaussian_priors[2][0] == 3.0
        assert gaussian_priors[3][0] == 4.1

        assert gaussian_priors[0][1] == pytest.approx(0.12, 1e-2)
        assert gaussian_priors[1][1] == pytest.approx(0.12, 1e-2)
        assert gaussian_priors[2][1] == pytest.approx(0.12, 1e-2)
        assert gaussian_priors[3][1] == pytest.approx(0.22, 1e-2)

    def test__1_profile__gaussian_priors_at_1_sigma_confidence(self, nlo_priors_path):

        conf.instance.output_path = nlo_priors_path

        mapper = model_mapper.ModelMapper(mock_class=MockClassNLOx4)
        nlo = MockNonLinearOptimizer(phase_name='', model_mapper=mapper, most_probable=[1.0, 2.0, 3.0, 4.1])
        create_weighted_samples_4_parameters(path=nlo.backup_path)

        gaussian_priors = nlo.gaussian_priors_at_sigma_limit(sigma_limit=1.0)

        assert gaussian_priors[0][0] == 1.0
        assert gaussian_priors[1][0] == 2.0
        assert gaussian_priors[2][0] == 3.0
        assert gaussian_priors[3][0] == 4.1

        assert gaussian_priors[0][1] == pytest.approx(1.0 - 0.927, 5e-2)
        assert gaussian_priors[1][1] == pytest.approx(2.0 - 1.928, 5e-2)
        assert gaussian_priors[2][1] == pytest.approx(3.0 - 2.928, 5e-2)
        assert gaussian_priors[3][1] == pytest.approx(4.1 - 3.928, 5e-2)


class TestOffsetFromInput:

    def test__input_model_offset_from_most_probable__parameters_and_instance__1_class_4_params(self):

        mapper = model_mapper.ModelMapper(mock_class=MockClassNLOx4)
        nlo = MockNonLinearOptimizer(phase_name='', model_mapper=mapper,
                                     most_probable=[1.0, -2.0, 3.0, 4.0])


        offset_values = nlo.offset_values_from_input_model_parameters(input_model_parameters=[1.0, 1.0, 2.0, 3.0])

        assert offset_values == [0.0, -3.0, 1.0, 1.0]

        mapper = model_mapper.ModelMapper(mock_class_1=MockClassNLOx4,
                                          mock_class_2=MockClassNLOx6)
        nlo = MockNonLinearOptimizer(phase_name='', model_mapper=mapper,
                                     most_probable=[1.0, 2.0, 3.0, 4.0, -5.0, -6.0, -7.0, -8.0, 9.0, 10.0])

        offset_values = nlo.offset_values_from_input_model_parameters(
            input_model_parameters=[1.0, 1.0, 2.0, 3.0, 10.0, 10.0, 10.0, 10.0, 10.0, 20.0])

        assert offset_values == [0.0, 1.0, 1.0, 1.0, -15.0, -16.0, -17.0, -18.0, -1.0, -10.0]


@pytest.fixture(name='optimizer')
def make_optimizer():
    return autofit.optimize.non_linear.non_linear.NonLinearOptimizer(phase_name='', )


class TestLabels(object):

    def test_param_names(self, optimizer):
        optimizer.variable.prior_model = MockClassNLOx4
        assert ['prior_model_one', 'prior_model_two', 'prior_model_three',
                'prior_model_four'] == optimizer.variable.param_names

    def test_properties(self, optimizer):
        optimizer.variable.prior_model = MockClassNLOx4

        assert len(optimizer.param_labels) == 4
        assert len(optimizer.variable.param_names) == 4

    def test_label_config(self):
        assert conf.instance.label.label("one") == "x4p0"
        assert conf.instance.label.label("two") == "x4p1"
        assert conf.instance.label.label("three") == "x4p2"
        assert conf.instance.label.label("four") == "x4p3"

    def test_labels(self, optimizer):
        autofit.mapper.prior_model.AbstractPriorModel._ids = itertools.count()
        optimizer.variable.prior_model = MockClassNLOx4

        assert optimizer.param_labels == [r'x4p0_{\mathrm{a2}}', r'x4p1_{\mathrm{a2}}',
                                          r'x4p2_{\mathrm{a2}}', r'x4p3_{\mathrm{a2}}']


class TestLatex(object):

    def test__results_for_