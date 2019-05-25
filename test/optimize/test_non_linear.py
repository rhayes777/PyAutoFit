import itertools
import os
import shutil
from functools import wraps

import pytest

import autofit.mapper.prior_model
import autofit.optimize.non_linear.downhill_simplex
import autofit.optimize.non_linear.grid_search
import autofit.optimize.non_linear.multi_nest
import autofit.optimize.non_linear.non_linear
from autofit import conf
from autofit import exc
from autofit import mock
from autofit.mapper import model_mapper
from test.mock.mock import MockClassNLOx4, MockClassNLOx5, MockClassNLOx6, MockAnalysis

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

    def test_downhill_simplex(self):
        optimizer = autofit.optimize.non_linear.downhill_simplex.DownhillSimplex("phase_name", fmin=lambda x: x)

        copy = optimizer.copy_with_name_extension("one")
        self.assert_non_linear_attributes_equal(copy, optimizer)
        assert isinstance(copy, autofit.optimize.non_linear.downhill_simplex.DownhillSimplex)
        assert copy.fmin is optimizer.fmin
        assert copy.xtol is optimizer.xtol
        assert copy.ftol is optimizer.ftol
        assert copy.maxiter is optimizer.maxiter
        assert copy.maxfun is optimizer.maxfun
        assert copy.full_output is optimizer.full_output
        assert copy.disp is optimizer.disp
        assert copy.retall is optimizer.retall

    def test_multinest(self):
        optimizer = autofit.optimize.non_linear.multi_nest.MultiNest("phase_name", sigma_limit=2.0, run=lambda x: x)

        copy = optimizer.copy_with_name_extension("one")
        self.assert_non_linear_attributes_equal(copy, optimizer)
        assert isinstance(copy, autofit.optimize.non_linear.multi_nest.MultiNest)
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


class TestNonLinearOptimizer(object):

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


@pytest.fixture(name="downhill_simplex")
def make_downhill_simplex():
    def fmin(fitness_function, x0):
        fitness_function(x0)
        return x0

    return autofit.optimize.non_linear.downhill_simplex.DownhillSimplex(fmin=fmin, phase_name='', model_mapper=model_mapper.ModelMapper())


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
