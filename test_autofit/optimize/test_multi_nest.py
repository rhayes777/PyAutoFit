import os
import shutil
from functools import wraps

import pytest

import autofit as af
from autofit import Paths
from autofit.optimize.non_linear.multi_nest import MultiNestOutput
from test_autofit.mock import MockClassNLOx4, MockClassNLOx6

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(name="multi_nest_summary_path")
def test_multi_nest_summary():
    multi_nest_summary_path = "{}/../test_files/non_linear/multinest/summary".format(
        os.path.dirname(os.path.realpath(__file__))
    )

    if os.path.exists(multi_nest_summary_path):
        shutil.rmtree(multi_nest_summary_path)

    os.mkdir(multi_nest_summary_path)

    return multi_nest_summary_path


@pytest.fixture(name="multi_nest_samples_path")
def test_multi_nest_samples():
    multi_nest_samples_path = "{}/../test_files/non_linear/multinest/samples".format(
        os.path.dirname(os.path.realpath(__file__))
    )

    if os.path.exists(multi_nest_samples_path):
        shutil.rmtree(multi_nest_samples_path)

    os.mkdir(multi_nest_samples_path)

    return multi_nest_samples_path


@pytest.fixture(name="multi_nest_phys_live_path")
def test_multi_nest_phys_live():
    multi_nest_phys_live_path = "{}/../test_files/non_linear/multinest/phys_live".format(
        os.path.dirname(os.path.realpath(__file__))
    )

    if os.path.exists(multi_nest_phys_live_path):
        shutil.rmtree(multi_nest_phys_live_path)

    os.mkdir(multi_nest_phys_live_path)

    return multi_nest_phys_live_path


@pytest.fixture(name="multi_nest_resume_path")
def test_multi_nest_resume():
    multi_nest_resume_path = "{}/../test_files/non_linear/multinest/resume".format(
        os.path.dirname(os.path.realpath(__file__))
    )

    if os.path.exists(multi_nest_resume_path):
        shutil.rmtree(multi_nest_resume_path)

    os.mkdir(multi_nest_resume_path)

    return multi_nest_resume_path


def create_path(func):
    @wraps(func)
    def wrapper(path):
        if not os.path.exists(path):
            os.makedirs(path)
        return func(path)

    return wrapper


@create_path
def create_summary_4_parameters(path):
    summary = open(path + "/multinestsummary.txt", "w")
    summary.write(
        "    0.100000000000000000E+01   -0.200000000000000000E+01    0.300000000000000000E+01"
        "    0.400000000000000000E+01   -0.500000000000000000E+01    0.600000000000000000E+01"
        "    0.700000000000000000E+01    0.800000000000000000E+01"
        "    0.900000000000000000E+01   -1.000000000000000000E+01   -1.100000000000000000E+01"
        "    1.200000000000000000E+01    1.300000000000000000E+01   -1.400000000000000000E+01"
        "   -1.500000000000000000E+01    1.600000000000000000E+01"
        "    0.020000000000000000E+00    0.999999990000000000E+07"
        "    0.020000000000000000E+00    0.999999990000000000E+07\n"
    )
    summary.write(
        "    0.100000000000000000E+01   -0.200000000000000000E+01    0.300000000000000000E+01"
        "    0.400000000000000000E+01   -0.500000000000000000E+01    0.600000000000000000E+01"
        "    0.700000000000000000E+01    0.800000000000000000E+01"
        "    0.900000000000000000E+01   -1.000000000000000000E+01   -1.100000000000000000E+01"
        "    1.200000000000000000E+01    1.300000000000000000E+01   -1.400000000000000000E+01"
        "   -1.500000000000000000E+01    1.600000000000000000E+01"
        "    0.020000000000000000E+00    0.999999990000000000E+07"
    )
    summary.close()


@create_path
def create_summary_10_parameters(path):
    summary = open(path + "/multinestsummary.txt", "w")
    summary.write(
        "    0.100000000000000000E+01    0.200000000000000000E+01    0.300000000000000000E+01"
        "    0.400000000000000000E+01   -0.500000000000000000E+01   -0.600000000000000000E+01"
        "   -0.700000000000000000E+01   -0.800000000000000000E+01    0.900000000000000000E+01"
        "    1.000000000000000000E+01    1.100000000000000000E+01    1.200000000000000000E+01"
        "    1.300000000000000000E+01    1.400000000000000000E+01    1.500000000000000000E+01"
        "    1.600000000000000000E+01   -1.700000000000000000E+01   -1.800000000000000000E+01"
        "    1.900000000000000000E+01    2.000000000000000000E+01    2.100000000000000000E+01"
        "    2.200000000000000000E+01    2.300000000000000000E+01    2.400000000000000000E+01"
        "    2.500000000000000000E+01   -2.600000000000000000E+01   -2.700000000000000000E+01"
        "    2.800000000000000000E+01    2.900000000000000000E+01    3.000000000000000000E+01"
        "    3.100000000000000000E+01    3.200000000000000000E+01    3.300000000000000000E+01"
        "    3.400000000000000000E+01   -3.500000000000000000E+01   -3.600000000000000000E+01"
        "    3.700000000000000000E+01   -3.800000000000000000E+01   -3.900000000000000000E+01"
        "    4.000000000000000000E+01"
        "    0.020000000000000000E+00    0.999999990000000000E+07"
        "    0.020000000000000000E+00    0.999999990000000000E+07\n"
    )
    summary.write(
        "    0.100000000000000000E+01    0.200000000000000000E+01    0.300000000000000000E+01"
        "    0.400000000000000000E+01   -0.500000000000000000E+01   -0.600000000000000000E+01"
        "   -0.700000000000000000E+01   -0.800000000000000000E+01    0.900000000000000000E+01"
        "    1.000000000000000000E+01    1.100000000000000000E+01    1.200000000000000000E+01"
        "    1.300000000000000000E+01    1.400000000000000000E+01    1.500000000000000000E+01"
        "    1.600000000000000000E+01   -1.700000000000000000E+01   -1.800000000000000000E+01"
        "    1.900000000000000000E+01    2.000000000000000000E+01    2.100000000000000000E+01"
        "    2.200000000000000000E+01    2.300000000000000000E+01    2.400000000000000000E+01"
        "    2.500000000000000000E+01   -2.600000000000000000E+01   -2.700000000000000000E+01"
        "    2.800000000000000000E+01    2.900000000000000000E+01    3.000000000000000000E+01"
        "    3.100000000000000000E+01    3.200000000000000000E+01    3.300000000000000000E+01"
        "    3.400000000000000000E+01   -3.500000000000000000E+01   -3.600000000000000000E+01"
        "    3.700000000000000000E+01   -3.800000000000000000E+01   -3.900000000000000000E+01"
        "    4.000000000000000000E+01"
        "    0.020000000000000000E+00    0.999999990000000000E+07"
    )
    summary.close()


@create_path
def create_weighted_samples_4_parameters(path):
    with open(path + "/multinest.txt", "w+") as weighted_samples:
        weighted_samples.write(
            "    0.020000000000000000E+00    0.999999990000000000E+07    0.110000000000000000E+01    "
            "0.210000000000000000E+01    0.310000000000000000E+01    0.410000000000000000E+01\n"
            "    0.020000000000000000E+00    0.999999990000000000E+07    0.090000000000000000E+01    "
            "0.190000000000000000E+01    0.290000000000000000E+01    0.390000000000000000E+01\n"
            "    0.010000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    "
            "0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01\n"
            "    0.050000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    "
            "0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01\n"
            "    0.100000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    "
            "0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01\n"
            "    0.100000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    "
            "0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01\n"
            "    0.100000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    "
            "0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01\n"
            "    0.100000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    "
            "0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01\n"
            "    0.200000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    "
            "0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01\n"
            "    0.300000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    "
            "0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01"
        )


@create_path
def create_phys_live_4_parameters(path):
    with open(path + "/multinestphys_live.points", "w+") as phys_live:
        phys_live.write(
            "    0.110000000000000000E+01    "
            "0.210000000000000000E+01    0.310000000000000000E+01    0.410000000000000000E+01    0.020000000000000000E+00   1\n"
            "    0.090000000000000000E+01    "
            "0.190000000000000000E+01    0.290000000000000000E+01    0.390000000000000000E+01    0.020000000000000000E+00   1\n"
            "    0.100000000000000000E+01    "
            "0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01    0.020000000000000000E+00   1\n"
            "    0.100000000000000000E+01    "
            "0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01    0.020000000000000000E+00   1\n"
            "    0.100000000000000000E+01    "
            "0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01    0.020000000000000000E+00   1\n"
            "    0.100000000000000000E+01    "
            "0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01    0.020000000000000000E+00   1\n"
            "    0.100000000000000000E+01    "
            "0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01    0.020000000000000000E+00   1\n"
            "    0.100000000000000000E+01    "
            "0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01    0.020000000000000000E+00   1\n"
            "    0.100000000000000000E+01    "
            "0.200000000000000000E+01    0.300000000000000000E+01    0.500000000000000000E+01    0.040000000000000000E+00   1\n"
            "    0.100000000000000000E+01    "
            "0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01    0.020000000000000000E+00   1"
        )



@create_path
def create_resume(path):
    with open(path + "/multinestresume.dat", "w+") as resume:
        resume.write(
            " F\n"
            "        3000       12345           1          50\n"
            "    0.502352236277967168E+05    0.502900436569068333E+05\n"
            " T\n"
            "   0\n"
            " T F     0          50\n"
            "    0.648698272260014622E-26    0.502352236277967168E+05    0.502900436569068333E+05\n")



@create_path
def create_resume_2(path):
    with open(path + "/multinestresume.dat", "w+") as resume:
        resume.write(
            " F\n"
            "          60          60           1          50\n"
            "    0.502352236277967168E+05    0.502900436569068333E+05\n"
            " T\n"
            "   0\n"
            " T F     0          50\n"
            "    0.648698272260014622E-26    0.502352236277967168E+05    0.502900436569068333E+05\n")

class TestMultiNestOutputConverged:
    def test__maximum_log_likelihood_and_evidence__from_summary(
        self, multi_nest_summary_path
    ):

        af.conf.instance.output_path = multi_nest_summary_path + "/2_classes"

        model = af.ModelMapper(mock_class_1=MockClassNLOx4, mock_class_2=MockClassNLOx6)
        multinest_output = MultiNestOutput(model, Paths())

        create_summary_10_parameters(path=multinest_output.paths.backup_path)

        assert multinest_output.maximum_log_likelihood == pytest.approx(
            9999999.9, 1.0e-4
        )
        assert multinest_output.evidence == pytest.approx(0.02, 1.0e-4)

    def test__most_probable_vector__from_summary(self, multi_nest_summary_path):
        af.conf.instance.output_path = multi_nest_summary_path + "/2_classes"

        model = af.ModelMapper(mock_class_1=MockClassNLOx4, mock_class_2=MockClassNLOx6)
        multinest_output = MultiNestOutput(model, Paths())

        create_summary_10_parameters(path=multinest_output.paths.backup_path)

        assert multinest_output.most_probable_vector == [
            1.0,
            2.0,
            3.0,
            4.0,
            -5.0,
            -6.0,
            -7.0,
            -8.0,
            9.0,
            10.0,
        ]

    def test__most_likely_vector__from_summary(self, multi_nest_summary_path):
        af.conf.instance.output_path = multi_nest_summary_path + "/2_classes"

        model = af.ModelMapper(mock_class_1=MockClassNLOx4, mock_class_2=MockClassNLOx6)
        multinest_output = MultiNestOutput(model, Paths())

        create_summary_10_parameters(path=multinest_output.paths.backup_path)

        assert multinest_output.most_likely_vector == [
            21.0,
            22.0,
            23.0,
            24.0,
            25.0,
            -26.0,
            -27.0,
            28.0,
            29.0,
            30.0,
        ]

    def test__vector_at_sigma__from_weighted_samples(self, multi_nest_samples_path):
        af.conf.instance.output_path = multi_nest_samples_path + "/1_class"

        model = af.ModelMapper(mock_class=MockClassNLOx4)
        multinest_output = MultiNestOutput(model, Paths())
        create_weighted_samples_4_parameters(path=multinest_output.paths.backup_path)
        multinest_output.create_paramnames_file()
        shutil.copy(
            src=multinest_output.paths.file_param_names,
            dst=multinest_output.paths.backup_path + "/multinest.paramnames",
        )

        params = multinest_output.vector_at_sigma(sigma=3.0)
        assert params[0][0:2] == pytest.approx((0.88, 1.12), 1e-2)
        assert params[1][0:2] == pytest.approx((1.88, 2.12), 1e-2)
        assert params[2][0:2] == pytest.approx((2.88, 3.12), 1e-2)
        assert params[3][0:2] == pytest.approx((3.88, 4.12), 1e-2)

        params = multinest_output.vector_at_sigma(sigma=1.0)
        assert params[0][0:2] == pytest.approx((0.93, 1.07), 1e-2)
        assert params[1][0:2] == pytest.approx((1.93, 2.07), 1e-2)
        assert params[2][0:2] == pytest.approx((2.93, 3.07), 1e-2)
        assert params[3][0:2] == pytest.approx((3.93, 4.07), 1e-2)

    def test__samples__model_parameters_weight_and_likelihood_from_sample_index__from_weighted_samples(
        self, multi_nest_samples_path
    ):
        af.conf.instance.output_path = multi_nest_samples_path + "/1_class"

        model = af.ModelMapper(mock_class=MockClassNLOx4)
        multinest_output = MultiNestOutput(model, Paths())
        create_weighted_samples_4_parameters(path=multinest_output.paths.backup_path)
        multinest_output.create_paramnames_file()
        shutil.copy(
            src=multinest_output.paths.file_param_names,
            dst=multinest_output.paths.backup_path + "/multinest.paramnames",
        )

        model = multinest_output.vector_from_sample_index(sample_index=0)
        weight = multinest_output.weight_from_sample_index(sample_index=0)
        likelihood = multinest_output.likelihood_from_sample_index(sample_index=0)

        assert model == [1.1, 2.1, 3.1, 4.1]
        assert weight == 0.02
        assert likelihood == -0.5 * 9999999.9

        model = multinest_output.vector_from_sample_index(sample_index=5)
        weight = multinest_output.weight_from_sample_index(sample_index=5)
        likelihood = multinest_output.likelihood_from_sample_index(sample_index=5)

        assert model == [1.0, 2.0, 3.0, 4.0]
        assert weight == 0.1
        assert likelihood == -0.5 * 9999999.9

    def test__total_samples__accepted_samples__acceptance_ratio__from_resume_file(self, multi_nest_resume_path):

        af.conf.instance.output_path = multi_nest_resume_path + "/2_classes"

        model = af.ModelMapper(mock_class_1=MockClassNLOx4, mock_class_2=MockClassNLOx6)
        multinest_output = MultiNestOutput(model, Paths())

        create_resume(path=multinest_output.paths.backup_path)

        assert multinest_output.accepted_samples == 3000
        assert multinest_output.total_samples == 12345
        assert multinest_output.acceptance_ratio == 3000 / 12345

        create_resume_2(path=multinest_output.paths.backup_path)

        assert multinest_output.accepted_samples == 60
        assert multinest_output.total_samples == 60
        assert multinest_output.acceptance_ratio == 1.0


class TestMultiNestOutputUnconverged:
    def test__maximum_log_likelihood_and_evidence__from_phys_live_points(
        self, multi_nest_phys_live_path
    ):

        af.conf.instance.output_path = multi_nest_phys_live_path + "/1_class"

        model = af.ModelMapper(mock_class_1=MockClassNLOx4)
        multinest_output = MultiNestOutput(model, Paths())

        create_phys_live_4_parameters(path=multinest_output.paths.backup_path)

        assert multinest_output.maximum_log_likelihood == pytest.approx(0.04, 1.0e-4)
        assert multinest_output.evidence == None

    def test__most_probable_vector__use_most_likely_if_no_summary(
        self, multi_nest_phys_live_path
    ):
        af.conf.instance.output_path = multi_nest_phys_live_path + "/1_class"

        model = af.ModelMapper(mock_class_1=MockClassNLOx4)
        multinest_output = MultiNestOutput(model, Paths())

        create_phys_live_4_parameters(path=multinest_output.paths.backup_path)

        assert multinest_output.most_probable_vector == [1.0, 2.0, 3.0, 5.0]

    def test__most_likely_vector__from_phys_live_points(
        self, multi_nest_phys_live_path
    ):
        af.conf.instance.output_path = multi_nest_phys_live_path + "/1_class"

        model = af.ModelMapper(mock_class_1=MockClassNLOx4)
        multinest_output = MultiNestOutput(model, Paths())

        create_phys_live_4_parameters(path=multinest_output.paths.backup_path)

        assert multinest_output.most_likely_vector == [1.0, 2.0, 3.0, 5.0]

    def test__vector_at_sigma__uses_min_max_of_phys_lives(
        self, multi_nest_phys_live_path
    ):
        af.conf.instance.output_path = multi_nest_phys_live_path + "/1_class"

        model = af.ModelMapper(mock_class=MockClassNLOx4)
        multinest_output = MultiNestOutput(model, Paths())
        create_phys_live_4_parameters(path=multinest_output.paths.backup_path)

        params = multinest_output.vector_at_sigma(sigma=3.0)
        assert params[0][0:2] == pytest.approx((0.9, 1.1), 1e-2)
        assert params[1][0:2] == pytest.approx((1.9, 2.1), 1e-2)
        assert params[2][0:2] == pytest.approx((2.9, 3.1), 1e-2)
        assert params[3][0:2] == pytest.approx((3.9, 5.0), 1e-2)

        params = multinest_output.vector_at_sigma(sigma=1.0)
        assert params[0][0:2] == pytest.approx((0.9, 1.1), 1e-2)
        assert params[1][0:2] == pytest.approx((1.9, 2.1), 1e-2)
        assert params[2][0:2] == pytest.approx((2.9, 3.1), 1e-2)
        assert params[3][0:2] == pytest.approx((3.9, 5.0), 1e-2)


@pytest.fixture(name="multi_nest")
def make_multi_nest():
    multi_nest_fit_path = "{}/test_fit".format(
        os.path.dirname(os.path.realpath(__file__))
    )

    try:
        shutil.rmtree(multi_nest_fit_path)
    except FileNotFoundError as e:
        print(e)

    af.conf.instance.output_path = multi_nest_fit_path

    # noinspection PyUnusedLocal,PyPep8Naming
    def run(
        fitness_function,
        prior,
        total_parameters,
        outputfiles_basename,
        n_clustering_params=None,
        wrapped_params=None,
        importance_nested_sampling=True,
        multimodal=True,
        const_efficiency_mode=False,
        n_live_points=400,
        evidence_tolerance=0.5,
        sampling_efficiency=0.8,
        n_iter_before_update=100,
        null_log_evidence=-1e90,
        max_modes=100,
        mode_tolerance=-1e90,
        seed=-1,
        verbose=False,
        resume=True,
        context=0,
        write_output=True,
        log_zero=-1e100,
        max_iter=0,
        init_MPI=False,
        dump_callback=None,
    ):

        fitness_function(
            [1 for _ in range(total_parameters)],
            total_parameters,
            total_parameters,
            None,
        )

    multi_nest = af.MultiNest(run=run, paths=Paths(phase_name="", remove_files=False))

    create_weighted_samples_4_parameters(multi_nest.paths.sym_path)
    create_summary_4_parameters(multi_nest.paths.sym_path)

    return multi_nest


class TestCopyWithNameExtension:
    @staticmethod
    def assert_non_linear_attributes_equal(copy):
        assert copy.paths.phase_name == "phase_name/one"

    def test_multinest(self):
        optimizer = af.MultiNest(Paths("phase_name"), sigma=2.0, run=lambda x: x)

        copy = optimizer.copy_with_name_extension("one")
        self.assert_non_linear_attributes_equal(copy)
        assert isinstance(copy, af.MultiNest)
        assert copy.sigma is optimizer.sigma
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
