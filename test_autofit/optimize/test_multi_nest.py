import os
import shutil
from functools import wraps

import pytest

import autofit as af
from autofit import Paths
from autofit.optimize.non_linear.multi_nest_output import MultiNestOutput
from test_autofit.mock import (
    MockClassNLOx4,
    MockClassNLOx5,
    MockClassNLOx6,
    MockAnalysis,
)

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(scope="session", autouse=True)
def do_something():
    af.conf.instance = af.conf.Config(
        "{}/../test_files/configs/non_linear".format(
            os.path.dirname(os.path.realpath(__file__))
        )
    )


@pytest.fixture(name="mn_summary_path")
def test_mn_summary():
    mn_summary_path = "{}/../test_files/non_linear/multinest/summary".format(
        os.path.dirname(os.path.realpath(__file__))
    )

    if os.path.exists(mn_summary_path):
        shutil.rmtree(mn_summary_path)

    os.mkdir(mn_summary_path)

    return mn_summary_path


@pytest.fixture(name="mn_priors_path")
def test_mn_priors():
    mn_priors_path = "{}/../test_files/non_linear/multinest/priors".format(
        os.path.dirname(os.path.realpath(__file__))
    )

    if os.path.exists(mn_priors_path):
        shutil.rmtree(mn_priors_path)

    os.mkdir(mn_priors_path)

    return mn_priors_path


@pytest.fixture(name="mn_samples_path")
def test_mn_samples():
    mn_samples_path = "{}/../test_files/non_linear/multinest/samples".format(
        os.path.dirname(os.path.realpath(__file__))
    )

    if os.path.exists(mn_samples_path):
        shutil.rmtree(mn_samples_path)

    os.mkdir(mn_samples_path)

    return mn_samples_path


@pytest.fixture(name="mn_results_path")
def test_mn_results():
    mn_results_path = "{}/../test_files/non_linear/multinest/results".format(
        os.path.dirname(os.path.realpath(__file__))
    )

    if os.path.exists(mn_results_path):
        shutil.rmtree(mn_results_path)

    return mn_results_path


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
def create_gaussian_prior_summary_4_parameters(path):
    summary = open(path + "/multinestsummary.txt", "w")
    summary.write(
        "    0.100000000000000000E+01    0.200000000000000000E+01    0.300000000000000000E+01"
        "    0.410000000000000000E+01    0.500000000000000000E+01    0.600000000000000000E+01"
        "    0.700000000000000000E+01    0.800000000000000000E+01"
        "    0.900000000000000000E+01    1.000000000000000000E+01    1.100000000000000000E+01"
        "    1.200000000000000000E+01    1.300000000000000000E+01    1.400000000000000000E+01"
        "    1.500000000000000000E+01    1.600000000000000000E+01"
        "    0.020000000000000000E+00    0.999999990000000000E+07"
        "    0.020000000000000000E+00    0.999999990000000000E+07\n"
    )
    summary.write(
        "    0.100000000000000000E+01    0.200000000000000000E+01    0.300000000000000000E+01"
        "    0.410000000000000000E+01    0.500000000000000000E+01    0.600000000000000000E+01"
        "    0.700000000000000000E+01    0.800000000000000000E+01"
        "    0.900000000000000000E+01    1.000000000000000000E+01    1.100000000000000000E+01"
        "    1.200000000000000000E+01    1.300000000000000000E+01    1.400000000000000000E+01"
        "    1.500000000000000000E+01    1.600000000000000000E+01"
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
def create_weighted_samples_10_parameters(path):
    weighted_samples = open(path + "/multinest.txt", "w")
    weighted_samples.write(
        "    0.020000000000000000E+00    0.999999990000000000E+07    0.110000000000000000E+01    "
        "0.210000000000000000E+01    0.310000000000000000E+01    0.410000000000000000E+01   "
        "-0.510000000000000000E+01   -0.610000000000000000E+01   -0.710000000000000000E+01   "
        "-0.810000000000000000E+01    0.910000000000000000E+01    1.010000000000000000E+01\n"
        "    0.020000000000000000E+00    0.999999990000000000E+07    0.090000000000000000E+01    "
        "0.190000000000000000E+01    0.290000000000000000E+01    0.390000000000000000E+01   "
        "-0.490000000000000000E+01   -0.590000000000000000E+01   -0.690000000000000000E+01   "
        "-0.790000000000000000E+01    0.890000000000000000E+01    0.990000000000000000E+01\n"
        "    0.010000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    "
        "0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01   "
        "-0.500000000000000000E+01   -0.600000000000000000E+01   -0.700000000000000000E+01   "
        "-0.800000000000000000E+01    0.900000000000000000E+01    1.000000000000000000E+01\n"
        "    0.050000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    "
        "0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01   "
        "-0.500000000000000000E+01   -0.600000000000000000E+01   -0.700000000000000000E+01   "
        "-0.800000000000000000E+01    0.900000000000000000E+01    1.000000000000000000E+01\n"
        "    0.100000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    "
        "0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01   "
        "-0.500000000000000000E+01   -0.600000000000000000E+01   -0.700000000000000000E+01   "
        "-0.800000000000000000E+01    0.900000000000000000E+01    1.000000000000000000E+01\n"
        "    0.100000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    "
        "0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01   "
        "-0.500000000000000000E+01   -0.600000000000000000E+01   -0.700000000000000000E+01   "
        "-0.800000000000000000E+01    0.900000000000000000E+01    1.000000000000000000E+01\n"
        "    0.100000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    "
        "0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01   "
        "-0.500000000000000000E+01   -0.600000000000000000E+01   -0.700000000000000000E+01   "
        "-0.800000000000000000E+01    0.900000000000000000E+01    1.000000000000000000E+01\n"
        "    0.100000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    "
        "0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01   "
        "-0.500000000000000000E+01   -0.600000000000000000E+01   -0.700000000000000000E+01   "
        "-0.800000000000000000E+01    0.900000000000000000E+01    1.000000000000000000E+01\n"
        "    0.200000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    "
        "0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01   "
        "-0.500000000000000000E+01   -0.600000000000000000E+01   -0.700000000000000000E+01   "
        "-0.800000000000000000E+01    0.900000000000000000E+01    1.000000000000000000E+01\n"
        "    0.300000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    "
        "0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01   "
        "-0.500000000000000000E+01   -0.600000000000000000E+01   -0.700000000000000000E+01   "
        "-0.800000000000000000E+01    0.900000000000000000E+01    1.000000000000000000E+01"
    )
    weighted_samples.close()


class TestNLOInheritance:
    def test__most_probable_parameters_and_instance(self, mn_summary_path):
        af.conf.instance.output_path = mn_summary_path + "/2_classes"

        mapper = af.ModelMapper(
            mock_class_1=MockClassNLOx4, mock_class_2=MockClassNLOx6
        )
        mn = MultiNestOutput(mapper, Paths())

        create_summary_10_parameters(path=mn.paths.backup_path)

        assert mn.most_probable_model_parameters == [
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

        most_probable = mn.most_probable_model_instance

        assert most_probable.mock_class_1.one == 1.0
        assert most_probable.mock_class_1.two == 2.0
        assert most_probable.mock_class_1.three == 3.0
        assert most_probable.mock_class_1.four == 4.0

        assert most_probable.mock_class_2.one == (-5.0, -6.0)
        assert most_probable.mock_class_2.two == (-7.0, -8.0)
        assert most_probable.mock_class_2.three == 9.0
        assert most_probable.mock_class_2.four == 10.0

        af.conf.instance.output_path = mn_summary_path + "/1_class"

        mapper = af.ModelMapper(mock_class=MockClassNLOx5)
        mapper.mock_class.five = 10.0

        mn = MultiNestOutput(mapper, Paths())
        create_summary_4_parameters(path=mn.paths.backup_path)

        most_probable = mn.most_probable_model_instance

        assert most_probable.mock_class.one == 1.0
        assert most_probable.mock_class.two == -2.0
        assert most_probable.mock_class.three == 3.0
        assert most_probable.mock_class.four == 4.0
        assert most_probable.mock_class.five == 10.0

    def test__most_likely_parameters_and_instance(self, mn_summary_path):
        af.conf.instance.output_path = mn_summary_path + "/2_classes"

        mapper = af.ModelMapper(
            mock_class_1=MockClassNLOx4, mock_class_2=MockClassNLOx6
        )
        mn = MultiNestOutput(mapper, Paths())

        create_summary_10_parameters(path=mn.paths.backup_path)

        assert mn.most_likely_model_parameters == [
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

        most_likely = mn.most_likely_model_instance

        assert most_likely.mock_class_1.one == 21.0
        assert most_likely.mock_class_1.two == 22.0
        assert most_likely.mock_class_1.three == 23.0
        assert most_likely.mock_class_1.four == 24.0

        assert most_likely.mock_class_2.one == (25.0, -26.0)
        assert most_likely.mock_class_2.two == (-27.0, 28.0)
        assert most_likely.mock_class_2.three == 29.0
        assert most_likely.mock_class_2.four == 30.0

        mapper = af.ModelMapper(mock_class=MockClassNLOx5)
        mapper.mock_class.five = 10.0
        mn = MultiNestOutput(mapper, Paths())
        create_summary_4_parameters(path=mn.paths.backup_path)

        most_likely = mn.most_likely_model_instance

        assert most_likely.mock_class.one == 9.0
        assert most_likely.mock_class.two == -10.0
        assert most_likely.mock_class.three == -11.0
        assert most_likely.mock_class.four == 12.0
        assert most_likely.mock_class.five == 10.0

    def test__gaussian_priors(self, mn_priors_path):
        af.conf.instance.output_path = mn_priors_path

        mapper = af.ModelMapper(mock_class=MockClassNLOx4)
        mn = MultiNestOutput(mapper, Paths())

        create_gaussian_prior_summary_4_parameters(path=mn.paths.backup_path)
        create_weighted_samples_4_parameters(path=mn.paths.backup_path)
        gaussian_priors = mn.gaussian_priors_at_sigma_limit(sigma_limit=3.0)

        assert gaussian_priors[0][0] == 1.0
        assert gaussian_priors[1][0] == 2.0
        assert gaussian_priors[2][0] == 3.0
        assert gaussian_priors[3][0] == 4.1

        assert gaussian_priors[0][1] == pytest.approx(0.12, 1e-2)
        assert gaussian_priors[1][1] == pytest.approx(0.12, 1e-2)
        assert gaussian_priors[2][1] == pytest.approx(0.12, 1e-2)
        assert gaussian_priors[3][1] == pytest.approx(0.22, 1e-2)

        gaussian_priors = mn.gaussian_priors_at_sigma_limit(sigma_limit=1.0)

        assert gaussian_priors[0][0] == 1.0
        assert gaussian_priors[1][0] == 2.0
        assert gaussian_priors[2][0] == 3.0
        assert gaussian_priors[3][0] == 4.1

        assert gaussian_priors[0][1] == pytest.approx(1.0 - 0.927, 5e-2)
        assert gaussian_priors[1][1] == pytest.approx(2.0 - 1.928, 5e-2)
        assert gaussian_priors[2][1] == pytest.approx(3.0 - 2.928, 5e-2)
        assert gaussian_priors[3][1] == pytest.approx(4.1 - 3.928, 5e-2)

    def test__offset_from_input(self, mn_summary_path):
        af.conf.instance.output_path = mn_summary_path + "/1_class"

        mapper = af.ModelMapper(mock_class=MockClassNLOx4)
        mn = MultiNestOutput(mapper, Paths())
        create_summary_4_parameters(path=mn.paths.backup_path)

        # mn.most_probable_model_parameters == [1.0, -2.0, 3.0, 4.0]

        offset_values = mn.offset_values_from_input_model_parameters(
            input_model_parameters=[1.0, 1.0, 2.0, 3.0]
        )

        assert offset_values == [0.0, -3.0, 1.0, 1.0]

        af.conf.instance.output_path = mn_summary_path + "/2_classes"

        mapper = af.ModelMapper(
            mock_class_1=MockClassNLOx4, mock_class_2=MockClassNLOx6
        )
        mn = MultiNestOutput(mapper, Paths())
        create_summary_10_parameters(path=mn.paths.backup_path)

        # mn.most_probable_model_parameters == [1.0, 2.0, 3.0, 4.0, -5.0, -6.0, -7.0, -8.0, 9.0, 10.0]

        offset_values = mn.offset_values_from_input_model_parameters(
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


class TestSamples(object):
    def test__1_class___model_parameters_instance_weight_and_likelihood(
        self, mn_samples_path
    ):
        af.conf.instance.output_path = mn_samples_path + "/1_class"

        mapper = af.ModelMapper(mock_class=MockClassNLOx4)
        mn = MultiNestOutput(mapper, Paths())
        create_weighted_samples_4_parameters(path=mn.paths.backup_path)

        model = mn.sample_model_parameters_from_sample_index(sample_index=0)
        instance = mn.sample_model_instance_from_sample_index(sample_index=0)
        weight = mn.sample_weight_from_sample_index(sample_index=0)
        likelihood = mn.sample_likelihood_from_sample_index(sample_index=0)

        assert mn.total_samples == 10
        assert model == [1.1, 2.1, 3.1, 4.1]
        assert instance.mock_class.one == 1.1
        assert instance.mock_class.two == 2.1
        assert instance.mock_class.three == 3.1
        assert instance.mock_class.four == 4.1
        assert weight == 0.02
        assert likelihood == -0.5 * 9999999.9

        model = mn.sample_model_parameters_from_sample_index(sample_index=5)
        instance = mn.sample_model_instance_from_sample_index(sample_index=5)
        weight = mn.sample_weight_from_sample_index(sample_index=5)
        likelihood = mn.sample_likelihood_from_sample_index(sample_index=5)

        assert mn.total_samples == 10
        assert model == [1.0, 2.0, 3.0, 4.0]
        assert instance.mock_class.one == 1.0
        assert instance.mock_class.two == 2.0
        assert instance.mock_class.three == 3.0
        assert instance.mock_class.four == 4.0
        assert weight == 0.1
        assert likelihood == -0.5 * 9999999.9

    def test__2_classes__model_parameters_instance_weight_and_likelihood(
        self, mn_samples_path
    ):
        af.conf.instance.output_path = mn_samples_path + "/2_classes"

        mapper = af.ModelMapper(
            mock_class_1=MockClassNLOx4, mock_class_2=MockClassNLOx6
        )
        mn = MultiNestOutput(mapper, Paths())
        create_weighted_samples_10_parameters(path=mn.paths.backup_path)

        model = mn.sample_model_parameters_from_sample_index(sample_index=0)
        instance = mn.sample_model_instance_from_sample_index(sample_index=0)
        weight = mn.sample_weight_from_sample_index(sample_index=0)
        likelihood = mn.sample_likelihood_from_sample_index(sample_index=0)

        assert mn.total_samples == 10
        assert model == [1.1, 2.1, 3.1, 4.1, -5.1, -6.1, -7.1, -8.1, 9.1, 10.1]
        assert instance.mock_class_1.one == 1.1
        assert instance.mock_class_1.two == 2.1
        assert instance.mock_class_1.three == 3.1
        assert instance.mock_class_1.four == 4.1
        assert instance.mock_class_2.one == (-5.1, -6.1)
        assert instance.mock_class_2.two == (-7.1, -8.1)
        assert instance.mock_class_2.three == 9.1
        assert instance.mock_class_2.four == 10.1
        assert weight == 0.02
        assert likelihood == -0.5 * 9999999.9

        model = mn.sample_model_parameters_from_sample_index(sample_index=5)
        instance = mn.sample_model_instance_from_sample_index(sample_index=5)
        weight = mn.sample_weight_from_sample_index(sample_index=5)
        likelihood = mn.sample_likelihood_from_sample_index(sample_index=5)

        assert mn.total_samples == 10
        assert model == [1.0, 2.0, 3.0, 4.0, -5.0, -6.0, -7.0, -8.0, 9.0, 10.0]
        assert instance.mock_class_1.one == 1.0
        assert instance.mock_class_1.two == 2.0
        assert instance.mock_class_1.three == 3.0
        assert instance.mock_class_1.four == 4.0
        assert instance.mock_class_2.one == (-5.0, -6.0)
        assert instance.mock_class_2.two == (-7.0, -8.0)
        assert instance.mock_class_2.three == 9.0
        assert instance.mock_class_2.four == 10.0
        assert weight == 0.1
        assert likelihood == -0.5 * 9999999.9


class TestLimits(object):
    def test__1_profile__limits_1d_vectors_via_weighted_samples__1d_vectors_are_correct(
        self, mn_samples_path
    ):
        af.conf.instance.output_path = mn_samples_path + "/1_class"

        mapper = af.ModelMapper(mock_class=MockClassNLOx4)
        mn = MultiNestOutput(mapper, Paths())
        create_weighted_samples_4_parameters(path=mn.paths.backup_path)

        params_upper = mn.model_parameters_at_upper_sigma_limit(sigma_limit=3.0)
        assert params_upper == pytest.approx([1.12, 2.12, 3.12, 4.12], 1e-2)
        params_lower = mn.model_parameters_at_lower_sigma_limit(sigma_limit=3.0)
        assert params_lower == pytest.approx([0.88, 1.88, 2.88, 3.88], 1e-2)

    def test__1_profile__change_limit_to_1_sigma(self, mn_samples_path):
        af.conf.instance.output_path = mn_samples_path + "/1_class"

        mapper = af.ModelMapper(mock_class=MockClassNLOx4)
        mn = MultiNestOutput(mapper, Paths())
        create_weighted_samples_4_parameters(path=mn.paths.backup_path)

        params_upper = mn.model_parameters_at_upper_sigma_limit(sigma_limit=1.0)
        assert params_upper == pytest.approx([1.07, 2.07, 3.07, 4.07], 1e-2)
        params_lower = mn.model_parameters_at_lower_sigma_limit(sigma_limit=1.0)
        assert params_lower == pytest.approx([0.93, 1.93, 2.93, 3.93], 1e-2)

    def test__1_species__errors_1d_vectors_via_weighted_samples__1d_vectors_are_correct(
        self, mn_samples_path
    ):
        af.conf.instance.output_path = mn_samples_path + "/1_class"

        mapper = af.ModelMapper(mock_class=MockClassNLOx4)
        mn = MultiNestOutput(mapper, Paths())
        create_weighted_samples_4_parameters(path=mn.paths.backup_path)

        model_errors = mn.model_errors_at_sigma_limit(sigma_limit=3.0)
        assert model_errors == pytest.approx(
            [1.12 - 0.88, 2.12 - 1.88, 3.12 - 2.88, 4.12 - 3.88], 1e-2
        )

        model_errors_instance = mn.model_errors_instance_at_sigma_limit(sigma_limit=3.0)
        assert model_errors_instance.mock_class.one == pytest.approx(1.12 - 0.88, 1e-2)
        assert model_errors_instance.mock_class.two == pytest.approx(2.12 - 1.88, 1e-2)
        assert model_errors_instance.mock_class.three == pytest.approx(
            3.12 - 2.88, 1e-2
        )
        assert model_errors_instance.mock_class.four == pytest.approx(4.12 - 3.88, 1e-2)

    def test__1_species__change_limit_to_1_sigma(self, mn_samples_path):
        af.conf.instance.output_path = mn_samples_path + "/1_class"

        mapper = af.ModelMapper(mock_class=MockClassNLOx4)
        mn = MultiNestOutput(mapper, Paths())
        create_weighted_samples_4_parameters(path=mn.paths.backup_path)

        model_errors = mn.model_errors_at_sigma_limit(sigma_limit=1.0)
        assert model_errors == pytest.approx(
            [1.07 - 0.93, 2.07 - 1.93, 3.07 - 2.93, 4.07 - 3.93], 1e-1
        )


@pytest.fixture(name="multi_nest")
def make_multi_nest():
    mn_fit_path = "{}/test_fit".format(os.path.dirname(os.path.realpath(__file__)))

    try:
        shutil.rmtree(mn_fit_path)
    except FileNotFoundError as e:
        print(e)

    af.conf.instance.output_path = mn_fit_path

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

    multi_nest = af.MultiNest(run=run, paths=Paths(phase_name=""))

    create_weighted_samples_4_parameters(multi_nest.paths.sym_path)
    create_summary_4_parameters(multi_nest.paths.sym_path)

    return multi_nest


class TestFitting(object):
    def test_model(self, multi_nest, model):
        model.mock_class = af.PriorModel(MockClassNLOx4)
        result = multi_nest.fit(MockAnalysis(), model)

        assert result.instance.mock_class.one == 9.0
        assert result.instance.mock_class.two == -10.0
        assert result.figure_of_merit == 0.02

        assert result.model.mock_class.one.mean == 1
        assert result.model.mock_class.two.mean == -2


class TestCopyWithNameExtension(object):
    @staticmethod
    def assert_non_linear_attributes_equal(copy):
        assert copy.paths.phase_name == "phase_name/one"

    def test_multinest(self):
        optimizer = af.MultiNest(Paths("phase_name"), sigma_limit=2.0, run=lambda x: x)

        copy = optimizer.copy_with_name_extension("one")
        self.assert_non_linear_attributes_equal(copy)
        assert isinstance(copy, af.MultiNest)
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
