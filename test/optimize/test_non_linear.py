import itertools
import os
import shutil
from functools import wraps

import pytest

import autofit.mapper.prior_model
from autofit import conf
from autofit import exc
from autofit import mock
from autofit.mapper import model_mapper, prior as p
from autofit.optimize import non_linear

pytestmark = pytest.mark.filterwarnings('ignore::FutureWarning')


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
    return non_linear.Result(None, None, mapper, [(0, 0), (1, 0)])


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
        optimizer = non_linear.NonLinearOptimizer("phase_name")
        copy = optimizer.copy_with_name_extension("one")

        self.assert_non_linear_attributes_equal(copy, optimizer)

    def test_downhill_simplex(self):
        optimizer = non_linear.DownhillSimplex("phase_name", fmin=lambda x: x)

        copy = optimizer.copy_with_name_extension("one")
        self.assert_non_linear_attributes_equal(copy, optimizer)
        assert isinstance(copy, non_linear.DownhillSimplex)
        assert copy.fmin is optimizer.fmin
        assert copy.xtol is optimizer.xtol
        assert copy.ftol is optimizer.ftol
        assert copy.maxiter is optimizer.maxiter
        assert copy.maxfun is optimizer.maxfun
        assert copy.full_output is optimizer.full_output
        assert copy.disp is optimizer.disp
        assert copy.retall is optimizer.retall

    def test_multinest(self):
        optimizer = non_linear.MultiNest("phase_name", sigma_limit=2.0, run=lambda x: x)

        copy = optimizer.copy_with_name_extension("one")
        self.assert_non_linear_attributes_equal(copy, optimizer)
        assert isinstance(copy, non_linear.MultiNest)
        assert copy.sigma_limit is optimizer.sigma_limit
        assert copy.run is optimizer.run
        assert copy.importance_nested_sampling is optimizer.importance_nested_sampling
        assert copy.multimodal is optimizer.multimodal
        assert copy.const_efficiency_mode is optimizer.const_efficiency_mode
        assert copy.n_live_points is optimizer.n_live_points
        assert copy.evidence_tolerance is optimizer.evidence_tolerance
        assert copy.sampling_efficiency is optimizer.sampling_efficiency
        assert copy.n_iter_before_update is optimizer.n_iter_before_update
        assert copy.null_log_evidence is optimizer.null_log_evidence
        assert copy.max_modes is optimizer.max_modes
        assert copy.mode_tolerance is optimizer.mode_tolerance
        assert copy.outputfiles_basename is optimizer.outputfiles_basename
        assert copy.seed is optimizer.seed
        assert copy.verbose is optimizer.verbose
        assert copy.resume is optimizer.resume
        assert copy.context is optimizer.context
        assert copy.write_output is optimizer.write_output
        assert copy.log_zero is optimizer.log_zero
        assert copy.max_iter is optimizer.max_iter
        assert copy.init_MPI is optimizer.init_MPI

    def test_grid_search(self):
        optimizer = non_linear.GridSearch("phase_name", step_size=17, grid=lambda x: x)

        copy = optimizer.copy_with_name_extension("one")
        self.assert_non_linear_attributes_equal(copy, optimizer)
        assert isinstance(copy, non_linear.GridSearch)
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


@pytest.fixture(name='mn_summary_path')
def test_mn_summary():
    mn_summary_path = "{}/../test_files/non_linear/multinest/summary".format(
        os.path.dirname(os.path.realpath(__file__)))

    if os.path.exists(mn_summary_path):
        shutil.rmtree(mn_summary_path)

    os.mkdir(mn_summary_path)

    return mn_summary_path


@pytest.fixture(name='mn_priors_path')
def test_mn_priors():
    mn_priors_path = "{}/../test_files/non_linear/multinest/priors".format(os.path.dirname(os.path.realpath(__file__)))

    if os.path.exists(mn_priors_path):
        shutil.rmtree(mn_priors_path)

    os.mkdir(mn_priors_path)

    return mn_priors_path


@pytest.fixture(name='mn_samples_path')
def test_mn_samples():
    mn_samples_path = "{}/../test_files/non_linear/multinest/samples".format(
        os.path.dirname(os.path.realpath(__file__)))

    if os.path.exists(mn_samples_path):
        shutil.rmtree(mn_samples_path)

    os.mkdir(mn_samples_path)

    return mn_samples_path


@pytest.fixture(name='mn_results_path')
def test_mn_results():
    mn_results_path = "{}/../test_files/non_linear/multinest/results".format(
        os.path.dirname(os.path.realpath(__file__)))

    if os.path.exists(mn_results_path):
        shutil.rmtree(mn_results_path)

    return mn_results_path


@pytest.fixture(scope="session", autouse=True)
def do_something():
    conf.instance = conf.Config(
        "{}/../test_files/configs/non_linear".format(os.path.dirname(os.path.realpath(__file__))))


def create_path(func):
    @wraps(func)
    def wrapper(path):
        if not os.path.exists(path):
            os.makedirs(path)
        return func(path)

    return wrapper


@create_path
def create_summary_4_parameters(path):
    summary = open(path + '/multinestsummary.txt', 'w')
    summary.write('    0.100000000000000000E+01   -0.200000000000000000E+01    0.300000000000000000E+01'
                  '    0.400000000000000000E+01   -0.500000000000000000E+01    0.600000000000000000E+01'
                  '    0.700000000000000000E+01    0.800000000000000000E+01'
                  '    0.900000000000000000E+01   -1.000000000000000000E+01   -1.100000000000000000E+01'
                  '    1.200000000000000000E+01    1.300000000000000000E+01   -1.400000000000000000E+01'
                  '   -1.500000000000000000E+01    1.600000000000000000E+01'
                  '    0.020000000000000000E+00    0.999999990000000000E+07'
                  '    0.020000000000000000E+00    0.999999990000000000E+07\n')
    summary.write('    0.100000000000000000E+01   -0.200000000000000000E+01    0.300000000000000000E+01'
                  '    0.400000000000000000E+01   -0.500000000000000000E+01    0.600000000000000000E+01'
                  '    0.700000000000000000E+01    0.800000000000000000E+01'
                  '    0.900000000000000000E+01   -1.000000000000000000E+01   -1.100000000000000000E+01'
                  '    1.200000000000000000E+01    1.300000000000000000E+01   -1.400000000000000000E+01'
                  '   -1.500000000000000000E+01    1.600000000000000000E+01'
                  '    0.020000000000000000E+00    0.999999990000000000E+07')
    summary.close()


@create_path
def create_summary_10_parameters(path):
    summary = open(path + '/multinestsummary.txt', 'w')
    summary.write('    0.100000000000000000E+01    0.200000000000000000E+01    0.300000000000000000E+01'
                  '    0.400000000000000000E+01   -0.500000000000000000E+01   -0.600000000000000000E+01'
                  '   -0.700000000000000000E+01   -0.800000000000000000E+01    0.900000000000000000E+01'
                  '    1.000000000000000000E+01    1.100000000000000000E+01    1.200000000000000000E+01'
                  '    1.300000000000000000E+01    1.400000000000000000E+01    1.500000000000000000E+01'
                  '    1.600000000000000000E+01   -1.700000000000000000E+01   -1.800000000000000000E+01'
                  '    1.900000000000000000E+01    2.000000000000000000E+01    2.100000000000000000E+01'
                  '    2.200000000000000000E+01    2.300000000000000000E+01    2.400000000000000000E+01'
                  '    2.500000000000000000E+01   -2.600000000000000000E+01   -2.700000000000000000E+01'
                  '    2.800000000000000000E+01    2.900000000000000000E+01    3.000000000000000000E+01'
                  '    3.100000000000000000E+01    3.200000000000000000E+01    3.300000000000000000E+01'
                  '    3.400000000000000000E+01   -3.500000000000000000E+01   -3.600000000000000000E+01'
                  '    3.700000000000000000E+01   -3.800000000000000000E+01   -3.900000000000000000E+01'
                  '    4.000000000000000000E+01'
                  '    0.020000000000000000E+00    0.999999990000000000E+07'
                  '    0.020000000000000000E+00    0.999999990000000000E+07\n')
    summary.write('    0.100000000000000000E+01    0.200000000000000000E+01    0.300000000000000000E+01'
                  '    0.400000000000000000E+01   -0.500000000000000000E+01   -0.600000000000000000E+01'
                  '   -0.700000000000000000E+01   -0.800000000000000000E+01    0.900000000000000000E+01'
                  '    1.000000000000000000E+01    1.100000000000000000E+01    1.200000000000000000E+01'
                  '    1.300000000000000000E+01    1.400000000000000000E+01    1.500000000000000000E+01'
                  '    1.600000000000000000E+01   -1.700000000000000000E+01   -1.800000000000000000E+01'
                  '    1.900000000000000000E+01    2.000000000000000000E+01    2.100000000000000000E+01'
                  '    2.200000000000000000E+01    2.300000000000000000E+01    2.400000000000000000E+01'
                  '    2.500000000000000000E+01   -2.600000000000000000E+01   -2.700000000000000000E+01'
                  '    2.800000000000000000E+01    2.900000000000000000E+01    3.000000000000000000E+01'
                  '    3.100000000000000000E+01    3.200000000000000000E+01    3.300000000000000000E+01'
                  '    3.400000000000000000E+01   -3.500000000000000000E+01   -3.600000000000000000E+01'
                  '    3.700000000000000000E+01   -3.800000000000000000E+01   -3.900000000000000000E+01'
                  '    4.000000000000000000E+01'
                  '    0.020000000000000000E+00    0.999999990000000000E+07')
    summary.close()


@create_path
def create_gaussian_prior_summary_4_parameters(path):
    summary = open(path + '/multinestsummary.txt', 'w')
    summary.write('    0.100000000000000000E+01    0.200000000000000000E+01    0.300000000000000000E+01'
                  '    0.410000000000000000E+01    0.500000000000000000E+01    0.600000000000000000E+01'
                  '    0.700000000000000000E+01    0.800000000000000000E+01'
                  '    0.900000000000000000E+01    1.000000000000000000E+01    1.100000000000000000E+01'
                  '    1.200000000000000000E+01    1.300000000000000000E+01    1.400000000000000000E+01'
                  '    1.500000000000000000E+01    1.600000000000000000E+01'
                  '    0.020000000000000000E+00    0.999999990000000000E+07'
                  '    0.020000000000000000E+00    0.999999990000000000E+07\n')
    summary.write('    0.100000000000000000E+01    0.200000000000000000E+01    0.300000000000000000E+01'
                  '    0.410000000000000000E+01    0.500000000000000000E+01    0.600000000000000000E+01'
                  '    0.700000000000000000E+01    0.800000000000000000E+01'
                  '    0.900000000000000000E+01    1.000000000000000000E+01    1.100000000000000000E+01'
                  '    1.200000000000000000E+01    1.300000000000000000E+01    1.400000000000000000E+01'
                  '    1.500000000000000000E+01    1.600000000000000000E+01'
                  '    0.020000000000000000E+00    0.999999990000000000E+07')
    summary.close()


@create_path
def create_weighted_samples_4_parameters(path):
    with open(path + '/multinest.txt', 'w+') as weighted_samples:
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


@create_path
def create_weighted_samples_10_parameters(path):
    weighted_samples = open(path + '/multinest.txt', 'w')
    weighted_samples.write(
        '    0.020000000000000000E+00    0.999999990000000000E+07    0.110000000000000000E+01    '
        '0.210000000000000000E+01    0.310000000000000000E+01    0.410000000000000000E+01   '
        '-0.510000000000000000E+01   -0.610000000000000000E+01   -0.710000000000000000E+01   '
        '-0.810000000000000000E+01    0.910000000000000000E+01    1.010000000000000000E+01\n'
        '    0.020000000000000000E+00    0.999999990000000000E+07    0.090000000000000000E+01    '
        '0.190000000000000000E+01    0.290000000000000000E+01    0.390000000000000000E+01   '
        '-0.490000000000000000E+01   -0.590000000000000000E+01   -0.690000000000000000E+01   '
        '-0.790000000000000000E+01    0.890000000000000000E+01    0.990000000000000000E+01\n'
        '    0.010000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    '
        '0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01   '
        '-0.500000000000000000E+01   -0.600000000000000000E+01   -0.700000000000000000E+01   '
        '-0.800000000000000000E+01    0.900000000000000000E+01    1.000000000000000000E+01\n'
        '    0.050000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    '
        '0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01   '
        '-0.500000000000000000E+01   -0.600000000000000000E+01   -0.700000000000000000E+01   '
        '-0.800000000000000000E+01    0.900000000000000000E+01    1.000000000000000000E+01\n'
        '    0.100000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    '
        '0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01   '
        '-0.500000000000000000E+01   -0.600000000000000000E+01   -0.700000000000000000E+01   '
        '-0.800000000000000000E+01    0.900000000000000000E+01    1.000000000000000000E+01\n'
        '    0.100000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    '
        '0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01   '
        '-0.500000000000000000E+01   -0.600000000000000000E+01   -0.700000000000000000E+01   '
        '-0.800000000000000000E+01    0.900000000000000000E+01    1.000000000000000000E+01\n'
        '    0.100000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    '
        '0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01   '
        '-0.500000000000000000E+01   -0.600000000000000000E+01   -0.700000000000000000E+01   '
        '-0.800000000000000000E+01    0.900000000000000000E+01    1.000000000000000000E+01\n'
        '    0.100000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    '
        '0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01   '
        '-0.500000000000000000E+01   -0.600000000000000000E+01   -0.700000000000000000E+01   '
        '-0.800000000000000000E+01    0.900000000000000000E+01    1.000000000000000000E+01\n'
        '    0.200000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    '
        '0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01   '
        '-0.500000000000000000E+01   -0.600000000000000000E+01   -0.700000000000000000E+01   '
        '-0.800000000000000000E+01    0.900000000000000000E+01    1.000000000000000000E+01\n'
        '    0.300000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    '
        '0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01   '
        '-0.500000000000000000E+01   -0.600000000000000000E+01   -0.700000000000000000E+01   '
        '-0.800000000000000000E+01    0.900000000000000000E+01    1.000000000000000000E+01')
    weighted_samples.close()


class MockClassNLOx4(object):

    def __init__(self, one=1, two=2, three=3, four=4):
        self.one = one
        self.two = two
        self.three = three
        self.four = four


class MockClassNLOx5(object):

    def __init__(self, one=1, two=2, three=3, four=4, five=5):
        self.one = one
        self.two = two
        self.three = three
        self.four = four
        self.five = five


class MockClassNLOx6(object):

    def __init__(self, one=(1, 2), two=(3, 4), three=3, four=4):
        self.one = one
        self.two = two
        self.three = three
        self.four = four


class TestNonLinearOptimizer(object):
    class TestDirectorySetup:

        def test__1_class__correct_directory(self, nlo_setup_path):
            conf.instance.output_path = nlo_setup_path + '1_class'
            mapper = model_mapper.ModelMapper(mock_class=MockClassNLOx4)
            non_linear.NonLinearOptimizer(phase_name='', model_mapper=mapper)

            assert os.path.exists(nlo_setup_path + '1_class')

    class TestTotalParameters:

        def test__1_class__four_parameters(self, nlo_setup_path):
            conf.instance.output_path = nlo_setup_path + '1_class'
            mapper = model_mapper.ModelMapper(mock_class=MockClassNLOx4)
            nlo = non_linear.NonLinearOptimizer(phase_name='', model_mapper=mapper)

            assert nlo.variable.prior_count == 4

        def test__2_classes__six_parameters(self, nlo_setup_path):
            conf.instance.output_path = nlo_setup_path + '2_classes'
            mapper = model_mapper.ModelMapper(class_1=MockClassNLOx4, class_2=MockClassNLOx6)
            nlo = non_linear.NonLinearOptimizer(phase_name='', model_mapper=mapper)

            assert nlo.variable.prior_count == 10


class TestMultiNest(object):
    class TestReadFromSummary:

        def test__most_probable_parameters_and_instance__1_class_4_params(self, mn_summary_path):
            conf.instance.output_path = mn_summary_path + '/1_class'

            mapper = model_mapper.ModelMapper(mock_class=MockClassNLOx4)
            mn = non_linear.MultiNest(phase_name='', model_mapper=mapper)
            create_summary_4_parameters(path=mn.opt_path)

            assert mn.most_probable_model_parameters == [1.0, -2.0, 3.0, 4.0]

            most_probable = mn.most_probable_model_instance

            assert most_probable.mock_class.one == 1.0
            assert most_probable.mock_class.two == -2.0
            assert most_probable.mock_class.three == 3.0
            assert most_probable.mock_class.four == 4.0

        def test__most_probable_parameters_and_instance__2_classes_6_params(self, mn_summary_path):
            conf.instance.output_path = mn_summary_path + '/2_classes'

            mapper = model_mapper.ModelMapper(mock_class_1=MockClassNLOx4,
                                              mock_class_2=MockClassNLOx6)
            mn = non_linear.MultiNest(phase_name='', model_mapper=mapper)
            create_summary_10_parameters(path=mn.opt_path)

            assert mn.most_probable_model_parameters == [1.0, 2.0, 3.0, 4.0, -5.0, -6.0, -7.0, -8.0, 9.0, 10.0]

            most_probable = mn.most_probable_model_instance

            assert most_probable.mock_class_1.one == 1.0
            assert most_probable.mock_class_1.two == 2.0
            assert most_probable.mock_class_1.three == 3.0
            assert most_probable.mock_class_1.four == 4.0

            assert most_probable.mock_class_2.one == (-5.0, -6.0)
            assert most_probable.mock_class_2.two == (-7.0, -8.0)
            assert most_probable.mock_class_2.three == 9.0
            assert most_probable.mock_class_2.four == 10.0

        def test__most_probable__setup_model_instance__1_class_5_params_but_1_is_constant(self, mn_summary_path):
            conf.instance.output_path = mn_summary_path + '/1_class'

            mapper = model_mapper.ModelMapper(mock_class=MockClassNLOx5)
            mapper.mock_class.five = p.Constant(10.0)

            mn = non_linear.MultiNest(phase_name='', model_mapper=mapper)
            create_summary_4_parameters(path=mn.opt_path)

            most_probable = mn.most_probable_model_instance

            assert most_probable.mock_class.one == 1.0
            assert most_probable.mock_class.two == -2.0
            assert most_probable.mock_class.three == 3.0
            assert most_probable.mock_class.four == 4.0
            assert most_probable.mock_class.five == 10.0

        def test__most_likely_parameters_and_instance__1_class_4_params(self, mn_summary_path):
            conf.instance.output_path = mn_summary_path + '/1_class'

            mapper = model_mapper.ModelMapper(mock_class=MockClassNLOx4)
            mn = non_linear.MultiNest(phase_name='', model_mapper=mapper)
            create_summary_4_parameters(path=mn.opt_path)

            assert mn.most_likely_model_parameters == [9.0, -10.0, -11.0, 12.0]

            most_likely = mn.most_likely_model_instance

            assert most_likely.mock_class.one == 9.0
            assert most_likely.mock_class.two == -10.0
            assert most_likely.mock_class.three == -11.0
            assert most_likely.mock_class.four == 12.0

        def test__rmost_likely_parameters_and_instance__2_classes_6_params(self, mn_summary_path):
            conf.instance.output_path = mn_summary_path + '/2_classes'

            mapper = model_mapper.ModelMapper(mock_class_1=MockClassNLOx4,
                                              mock_class_2=MockClassNLOx6)
            mn = non_linear.MultiNest(phase_name='', model_mapper=mapper)

            create_summary_10_parameters(path=mn.opt_path)

            assert mn.most_likely_model_parameters == [21.0, 22.0, 23.0, 24.0, 25.0, -26.0, -27.0, 28.0, 29.0, 30.0]

            most_likely = mn.most_likely_model_instance

            assert most_likely.mock_class_1.one == 21.0
            assert most_likely.mock_class_1.two == 22.0
            assert most_likely.mock_class_1.three == 23.0
            assert most_likely.mock_class_1.four == 24.0

            assert most_likely.mock_class_2.one == (25.0, -26.0)
            assert most_likely.mock_class_2.two == (-27.0, 28.0)
            assert most_likely.mock_class_2.three == 29.0
            assert most_likely.mock_class_2.four == 30.0

        def test__most_likely__setup_model_instance__1_class_5_params_but_1_is_constant(self, mn_summary_path):
            conf.instance.output_path = mn_summary_path + '/1_class'

            mapper = model_mapper.ModelMapper(mock_class=MockClassNLOx5)
            mapper.mock_class.five = p.Constant(10.0)
            mn = non_linear.MultiNest(phase_name='', model_mapper=mapper)
            create_summary_4_parameters(path=mn.opt_path)

            most_likely = mn.most_likely_model_instance

            assert most_likely.mock_class.one == 9.0
            assert most_likely.mock_class.two == -10.0
            assert most_likely.mock_class.three == -11.0
            assert most_likely.mock_class.four == 12.0
            assert most_likely.mock_class.five == 10.0

    class TestGaussianPriors(object):

        def test__1_class__gaussian_priors_at_3_sigma_confidence(self, mn_priors_path):
            conf.instance.output_path = mn_priors_path

            mapper = model_mapper.ModelMapper(mock_class=MockClassNLOx4)
            mn = non_linear.MultiNest(phase_name='', model_mapper=mapper)

            create_gaussian_prior_summary_4_parameters(path=mn.opt_path)
            create_weighted_samples_4_parameters(path=mn.opt_path)
            gaussian_priors = mn.gaussian_priors_at_sigma_limit(sigma_limit=3.0)

            assert gaussian_priors[0][0] == 1.0
            assert gaussian_priors[1][0] == 2.0
            assert gaussian_priors[2][0] == 3.0
            assert gaussian_priors[3][0] == 4.1

            assert gaussian_priors[0][1] == pytest.approx(0.12, 1e-2)
            assert gaussian_priors[1][1] == pytest.approx(0.12, 1e-2)
            assert gaussian_priors[2][1] == pytest.approx(0.12, 1e-2)
            assert gaussian_priors[3][1] == pytest.approx(0.22, 1e-2)

        def test__1_profile__gaussian_priors_at_1_sigma_confidence(self, mn_priors_path):
            conf.instance.output_path = mn_priors_path

            mapper = model_mapper.ModelMapper(mock_class=MockClassNLOx4)
            mn = non_linear.MultiNest(phase_name='', model_mapper=mapper)
            create_gaussian_prior_summary_4_parameters(path=mn.opt_path)
            create_weighted_samples_4_parameters(path=mn.opt_path)

            gaussian_priors = mn.gaussian_priors_at_sigma_limit(sigma_limit=1.0)

            # Use sigmas directly as rouding errors come in otherwise
            lower_sigmas = mn.model_parameters_at_lower_sigma_limit(sigma_limit=1.0)

            assert gaussian_priors[0][0] == 1.0
            assert gaussian_priors[1][0] == 2.0
            assert gaussian_priors[2][0] == 3.0
            assert gaussian_priors[3][0] == 4.1

            assert gaussian_priors[0][1] == pytest.approx(1.0 - lower_sigmas[0], 5e-2)
            assert gaussian_priors[1][1] == pytest.approx(2.0 - lower_sigmas[1], 5e-2)
            assert gaussian_priors[2][1] == pytest.approx(3.0 - lower_sigmas[2], 5e-2)
            assert gaussian_priors[3][1] == pytest.approx(4.1 - lower_sigmas[3], 5e-2)

    class TestWeightedSamples(object):

        def test__1_class__1st_first_weighted_sample__model_weight_and_likelihood(self, mn_samples_path):
            conf.instance.output_path = mn_samples_path + '/1_class'

            mapper = model_mapper.ModelMapper(mock_class=MockClassNLOx4)
            mn = non_linear.MultiNest(phase_name='', model_mapper=mapper)
            create_weighted_samples_4_parameters(path=mn.opt_path)

            model, weight, likelihood = mn.weighted_sample_model_parameters_from_weighted_samples(index=0)

            assert model == [1.1, 2.1, 3.1, 4.1]
            assert weight == 0.02
            assert likelihood == -0.5 * 9999999.9

        def test__1_class__5th_weighted_sample__model_weight_and_likelihood(self, mn_samples_path):
            conf.instance.output_path = mn_samples_path + '/1_class'

            mapper = model_mapper.ModelMapper(mock_class=MockClassNLOx4)
            mn = non_linear.MultiNest(phase_name='', model_mapper=mapper)
            create_weighted_samples_4_parameters(path=mn.opt_path)

            model, weight, likelihood = mn.weighted_sample_model_parameters_from_weighted_samples(index=5)

            assert model == [1.0, 2.0, 3.0, 4.0]
            assert weight == 0.1
            assert likelihood == -0.5 * 9999999.9

        def test__2_classes__1st_weighted_sample__model_weight_and_likelihood(self, mn_samples_path):
            conf.instance.output_path = mn_samples_path + '/2_classes'

            mapper = model_mapper.ModelMapper(mock_class_1=MockClassNLOx4,
                                              mock_class_2=MockClassNLOx6)
            mn = non_linear.MultiNest(phase_name='', model_mapper=mapper)
            create_weighted_samples_10_parameters(path=mn.opt_path)

            model, weight, likelihood = mn.weighted_sample_model_parameters_from_weighted_samples(index=0)

            assert model == [1.1, 2.1, 3.1, 4.1, -5.1, -6.1, -7.1, -8.1, 9.1, 10.1]
            assert weight == 0.02
            assert likelihood == -0.5 * 9999999.9

        def test__2_classes__5th_weighted_sample__model_weight_and_likelihood(self, mn_samples_path):
            conf.instance.output_path = mn_samples_path + '/2_classes'

            mapper = model_mapper.ModelMapper(mock_class_1=MockClassNLOx4,
                                              mock_class_2=MockClassNLOx6)
            mn = non_linear.MultiNest(phase_name='', model_mapper=mapper)
            create_weighted_samples_10_parameters(path=mn.opt_path)

            model, weight, likelihood = mn.weighted_sample_model_parameters_from_weighted_samples(index=5)

            assert model == [1.0, 2.0, 3.0, 4.0, -5.0, -6.0, -7.0, -8.0, 9.0, 10.0]
            assert weight == 0.1
            assert likelihood == -0.5 * 9999999.9

        def test__1_class__1st_weighted_sample_model_instance__include_weight_and_likelihood(self,
                                                                                             mn_samples_path):
            conf.instance.output_path = mn_samples_path + '/1_class'

            mapper = model_mapper.ModelMapper(mock_class=MockClassNLOx4)
            mn = non_linear.MultiNest(phase_name='', model_mapper=mapper)
            create_weighted_samples_4_parameters(path=mn.opt_path)

            weighted_sample_model, weight, likelihood = mn.weighted_sample_model_instance_from_weighted_samples(index=0)

            assert weight == 0.02
            assert likelihood == -0.5 * 9999999.9

            assert weighted_sample_model.mock_class.one == 1.1
            assert weighted_sample_model.mock_class.two == 2.1
            assert weighted_sample_model.mock_class.three == 3.1
            assert weighted_sample_model.mock_class.four == 4.1

        def test__1_class__5th_weighted_sample_model_instance__include_weight_and_likelihood(self,
                                                                                             mn_samples_path):
            conf.instance.output_path = mn_samples_path + '/1_class'

            mapper = model_mapper.ModelMapper(mock_class=MockClassNLOx4)
            mn = non_linear.MultiNest(phase_name='', model_mapper=mapper)
            create_weighted_samples_4_parameters(path=mn.opt_path)

            weighted_sample_model, weight, likelihood = mn.weighted_sample_model_instance_from_weighted_samples(index=5)

            assert weight == 0.1
            assert likelihood == -0.5 * 9999999.9

            assert weighted_sample_model.mock_class.one == 1.0
            assert weighted_sample_model.mock_class.two == 2.0
            assert weighted_sample_model.mock_class.three == 3.0
            assert weighted_sample_model.mock_class.four == 4.0

        def test__2_classes__1st_weighted_sample_model_instance__include_weight_and_likelihood(self,
                                                                                               mn_samples_path):
            conf.instance.output_path = mn_samples_path + '/2_classes'

            mapper = model_mapper.ModelMapper(mock_class_1=MockClassNLOx4,
                                              mock_class_2=MockClassNLOx6)
            mn = non_linear.MultiNest(phase_name='', model_mapper=mapper)
            create_weighted_samples_10_parameters(path=mn.opt_path)

            weighted_sample_model, weight, likelihood = mn.weighted_sample_model_instance_from_weighted_samples(index=0)

            assert weight == 0.02
            assert likelihood == -0.5 * 9999999.9

            assert weighted_sample_model.mock_class_1.one == 1.1
            assert weighted_sample_model.mock_class_1.two == 2.1
            assert weighted_sample_model.mock_class_1.three == 3.1
            assert weighted_sample_model.mock_class_1.four == 4.1

            assert weighted_sample_model.mock_class_2.one == (-5.1, -6.1)
            assert weighted_sample_model.mock_class_2.two == (-7.1, -8.1)
            assert weighted_sample_model.mock_class_2.three == 9.1
            assert weighted_sample_model.mock_class_2.four == 10.1

        def test__2_classes__5th_weighted_sample_model_instance__include_weight_and_likelihood(self,
                                                                                               mn_samples_path):
            conf.instance.output_path = mn_samples_path + '/2_classes'

            mapper = model_mapper.ModelMapper(mock_class_1=MockClassNLOx4,
                                              mock_class_2=MockClassNLOx6)
            mn = non_linear.MultiNest(phase_name='', model_mapper=mapper)
            create_weighted_samples_10_parameters(path=mn.opt_path)

            weighted_sample_model, weight, likelihood = mn.weighted_sample_model_instance_from_weighted_samples(index=5)

            assert weight == 0.1
            assert likelihood == -0.5 * 9999999.9

            assert weighted_sample_model.mock_class_1.one == 1.0
            assert weighted_sample_model.mock_class_1.two == 2.0
            assert weighted_sample_model.mock_class_1.three == 3.0
            assert weighted_sample_model.mock_class_1.four == 4.0

            assert weighted_sample_model.mock_class_2.one == (-5.0, -6.0)
            assert weighted_sample_model.mock_class_2.two == (-7.0, -8.0)
            assert weighted_sample_model.mock_class_2.three == 9.0
            assert weighted_sample_model.mock_class_2.four == 10.0

    class TestLimits(object):

        def test__1_profile__limits_1d_vectors_via_weighted_samples__1d_vectors_are_correct(self,
                                                                                            mn_samples_path):
            conf.instance.output_path = mn_samples_path + '/1_class'

            mapper = model_mapper.ModelMapper(mock_class=MockClassNLOx4)
            mn = non_linear.MultiNest(phase_name='', model_mapper=mapper)
            create_weighted_samples_4_parameters(path=mn.opt_path)

            params_upper = mn.model_parameters_at_upper_sigma_limit(sigma_limit=3.0)
            assert params_upper == pytest.approx([1.12, 2.12, 3.12, 4.12], 1e-2)
            params_lower = mn.model_parameters_at_lower_sigma_limit(sigma_limit=3.0)
            assert params_lower == pytest.approx([0.88, 1.88, 2.88, 3.88], 1e-2)

        def test__1_profile__change_limit_to_1_sigma(self, mn_samples_path):
            conf.instance.output_path = mn_samples_path + '/1_class'

            mapper = model_mapper.ModelMapper(mock_class=MockClassNLOx4)
            mn = non_linear.MultiNest(phase_name='', model_mapper=mapper)
            create_weighted_samples_4_parameters(path=mn.opt_path)

            params_upper = mn.model_parameters_at_upper_sigma_limit(sigma_limit=1.0)
            assert params_upper == pytest.approx([1.07, 2.07, 3.07, 4.07], 1e-2)
            params_lower = mn.model_parameters_at_lower_sigma_limit(sigma_limit=1.0)
            assert params_lower == pytest.approx([0.93, 1.93, 2.93, 3.93], 1e-2)

        # class TestModelErrors(object):
        #
        #     def test__1_species__upper_and_low_errors_1d_vectors_via_weighted_samples__1d_vectors_are_correct(
        #             self, mn_summary_path, mn_samples_path):
        #
        #         conf.instance.output_path = mn_samples_path + '/1_class'
        #
        #         mapper = model_mapper.ModelMapper(mock_class=MockClassNLOx4)
        #         mn = non_linear.MultiNest(phase_name='', model_mapper=mapper)
        #         create_summary_4_parameters(path=mn_summary_path)
        #         create_weighted_samples_4_parameters(path=mn_samples_path)
        #
        #         errors_upper = mn.model_errors_at_upper_sigma_limit(sigma_limit=3.0)
        #         assert errors_upper == pytest.approx([0.12, 0.12, 0.12, 0.12], 1e-2)
        #         errors_lower = mn.model_errors_at_lower_sigma_limit(sigma_limit=3.0)
        #         assert errors_lower == pytest.approx([0.12, 0.12, 0.12, 0.12], 1e-2)
        #
        #
        #     def test__1_species__upper_and_lower_change_limit_to_1_sigma(self, mn_samples_path):
        #
        #         conf.instance.output_path = mn_samples_path + '/1_class'
        #
        #         mapper = model_mapper.ModelMapper(mock_class=MockClassNLOx4)
        #         mn = non_linear.MultiNest(phase_name='', model_mapper=mapper)
        #         create_weighted_samples_4_parameters(path=mn_samples_path)
        #
        #         errors_upper = mn.model_errors_at_upper_sigma_limit(sigma_limit=1.0)
        #         assert errors_upper == pytest.approx([0.07, 0.07, 0.07, 0.07], 1e-2)
        #         errors_lower = mn.model_errors_at_lower_sigma_limit(sigma_limit=1.0)
        #         assert errors_lower == pytest.approx([0.07, 0.07, 0.07, 0.07], 1e-2)

        def test__1_species__errors_1d_vectors_via_weighted_samples__1d_vectors_are_correct(self,
                                                                                            mn_samples_path):
            conf.instance.output_path = mn_samples_path + '/1_class'

            mapper = model_mapper.ModelMapper(mock_class=MockClassNLOx4)
            mn = non_linear.MultiNest(phase_name='', model_mapper=mapper)
            create_weighted_samples_4_parameters(path=mn.opt_path)

            model_errors = mn.model_errors_at_sigma_limit(sigma_limit=3.0)
            assert model_errors == pytest.approx([1.12 - 0.88, 2.12 - 1.88, 3.12 - 2.88, 4.12 - 3.88], 1e-2)

        def test__1_species__change_limit_to_1_sigma(self, mn_samples_path):
            conf.instance.output_path = mn_samples_path + '/1_class'

            mapper = model_mapper.ModelMapper(mock_class=MockClassNLOx4)
            mn = non_linear.MultiNest(phase_name='', model_mapper=mapper)
            create_weighted_samples_4_parameters(path=mn.opt_path)

            model_errors = mn.model_errors_at_sigma_limit(sigma_limit=1.0)
            assert model_errors == pytest.approx([1.07 - 0.93, 2.07 - 1.93, 3.07 - 2.93, 4.07 - 3.93], 1e-1)

    class TestOffsetFromInput:

        def test__input_model_offset_from_most_probable__parameters_and_instance__1_class_4_params(self,
                                                                                                   mn_summary_path):
            conf.instance.output_path = mn_summary_path + '/1_class'

            mapper = model_mapper.ModelMapper(mock_class=MockClassNLOx4)
            mn = non_linear.MultiNest(phase_name='', model_mapper=mapper)
            create_summary_4_parameters(path=mn.opt_path)

            # mn.most_probable_model_parameters == [1.0, -2.0, 3.0, 4.0]

            offset_values = mn.offset_values_from_input_model_parameters(input_model_parameters=[1.0, 1.0, 2.0, 3.0])

            assert offset_values == [0.0, -3.0, 1.0, 1.0]

        def test__read_most_probable__2_classes_6_params(self, mn_summary_path):
            conf.instance.output_path = mn_summary_path + '/2_classes'

            mapper = model_mapper.ModelMapper(mock_class_1=MockClassNLOx4,
                                              mock_class_2=MockClassNLOx6)
            mn = non_linear.MultiNest(phase_name='', model_mapper=mapper)
            create_summary_10_parameters(path=mn.opt_path)

            # mn.most_probable_model_parameters == [1.0, 2.0, 3.0, 4.0, -5.0, -6.0, -7.0, -8.0, 9.0, 10.0]

            offset_values = mn.offset_values_from_input_model_parameters(
                input_model_parameters=[1.0, 1.0, 2.0, 3.0, 10.0, 10.0, 10.0, 10.0, 10.0, 20.0])

            assert offset_values == [0.0, 1.0, 1.0, 1.0, -15.0, -16.0, -17.0, -18.0, -1.0, -10.0]


class MockAnalysis(object):
    def __init__(self):
        self.kwargs = None
        self.instance = None
        self.visualise_instance = None

    def fit(self, instance):
        self.instance = instance
        return 1.

    # noinspection PyUnusedLocal
    def visualize(self, instance, *args, **kwargs):
        self.visualise_instance = instance


@pytest.fixture(name="downhill_simplex")
def make_downhill_simplex():
    def fmin(fitness_function, x0):
        fitness_function(x0)
        return x0

    return non_linear.DownhillSimplex(fmin=fmin, phase_name='', model_mapper=model_mapper.ModelMapper())


@pytest.fixture(name="multi_nest")
def make_multi_nest():
    mn_fit_path = "{}/test_fit".format(os.path.dirname(os.path.realpath(__file__)))
    try:
        shutil.rmtree(mn_fit_path)
    except FileNotFoundError as e:
        print(e)

    conf.instance.output_path = mn_fit_path

    # noinspection PyUnusedLocal,PyPep8Naming
    def run(fitness_function, prior, total_parameters, outputfiles_basename, n_clustering_params=None,
            wrapped_params=None, importance_nested_sampling=True, multimodal=True, const_efficiency_mode=False,
            n_live_points=400, evidence_tolerance=0.5, sampling_efficiency=0.8, n_iter_before_update=100,
            null_log_evidence=-1e+90, max_modes=100, mode_tolerance=-1e+90, seed=-1, verbose=False, resume=True,
            context=0,
            write_output=True, log_zero=-1e+100, max_iter=0, init_MPI=False, dump_callback=None):
        fitness_function([1 for _ in range(total_parameters)], total_parameters, total_parameters, None)

    multi_nest = non_linear.MultiNest(run=run, phase_name='', model_mapper=model_mapper.ModelMapper())

    create_weighted_samples_4_parameters(multi_nest.opt_path)
    create_summary_4_parameters(multi_nest.opt_path)

    return multi_nest


class TestFitting(object):
    class TestDownhillSimplex(object):
        def test_constant(self, downhill_simplex):
            downhill_simplex.variable.mock_class = MockClassNLOx4()

            assert len(downhill_simplex.variable.instance_tuples) == 1
            assert hasattr(downhill_simplex.variable.instance_from_unit_vector([]), "mock_class")

            result = downhill_simplex.fit(MockAnalysis())

            assert result.constant.mock_class.one == 1
            assert result.constant.mock_class.two == 2
            assert result.figure_of_merit == 1

        def test_variable(self, downhill_simplex):
            downhill_simplex.variable.mock_class = autofit.mapper.prior_model.PriorModel(MockClassNLOx4)
            result = downhill_simplex.fit(MockAnalysis())

            assert result.constant.mock_class.one == 0.0
            assert result.constant.mock_class.two == 0.0
            assert result.figure_of_merit == 1

            assert result.variable.mock_class.one.mean == 0.0
            assert result.variable.mock_class.two.mean == 0.0

        def test_constant_and_variable(self, downhill_simplex):
            downhill_simplex.variable.constant = MockClassNLOx4()
            downhill_simplex.variable.variable = autofit.mapper.prior_model.PriorModel(MockClassNLOx4)

            result = downhill_simplex.fit(MockAnalysis())

            assert result.constant.constant.one == 1
            assert result.constant.constant.two == 2
            assert result.constant.variable.one == 0.0
            assert result.constant.variable.two == 0.0
            assert result.variable.variable.one.mean == 0.0
            assert result.variable.variable.two.mean == 0.0
            assert result.figure_of_merit == 1

    class TestMultiNest(object):

        def test_variable(self, multi_nest):
            multi_nest.variable.mock_class = autofit.mapper.prior_model.PriorModel(MockClassNLOx4, )
            result = multi_nest.fit(MockAnalysis())

            assert result.constant.mock_class.one == 9.0
            assert result.constant.mock_class.two == -10.0
            assert result.figure_of_merit == 0.02

            assert result.variable.mock_class.one.mean == 1
            assert result.variable.mock_class.two.mean == -2


@pytest.fixture(name='optimizer')
def make_optimizer():
    return non_linear.NonLinearOptimizer(phase_name='', )


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
