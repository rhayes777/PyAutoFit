import os
import shutil
from functools import wraps
from os import path

import pytest

import autofit as af
from autoconf import conf
from autofit.mock import mock
from autofit.non_linear.nest.multinest import samples as samp

directory = path.dirname(path.realpath(__file__))
pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(name="multinest_summary_path")
def test_multinest_summary():
    multinest_summary_path = path.join(conf.instance.output_path, "non_linear", "multinest", "summary")

    if path.exists(multinest_summary_path):
        shutil.rmtree(multinest_summary_path)

    os.mkdir(multinest_summary_path)

    return multinest_summary_path


@pytest.fixture(name="multinest_samples_path")
def test_multinest_samples():
    multinest_samples_path = path.join(conf.instance.output_path, "non_linear", "multinest", "samples")

    if path.exists(multinest_samples_path):
        shutil.rmtree(multinest_samples_path)

    os.mkdir(multinest_samples_path)

    return multinest_samples_path


@pytest.fixture(name="multinest_resume_path")
def test_multinest_resume():
    multinest_resume_path = path.join(conf.instance.output_path, "non_linear", "multinest", "resume")

    if path.exists(multinest_resume_path):
        shutil.rmtree(multinest_resume_path)

    os.mkdir(multinest_resume_path)

    return multinest_resume_path


def create_path(func):
    @wraps(func)
    def wrapper(file_path):
        if not path.exists(file_path):
            os.makedirs(file_path)
        return func(file_path)

    return wrapper


@create_path
def create_summary_4_parameters(file_path):
    summary = open(path.join(file_path, "multinestsummary.txt"), "w")
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
def create_weighted_samples_4_parameters(file_path):
    with open(path.join(file_path, "multinest.txt"), "w+") as weighted_samples:
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
def create_resume(file_path):
    with open(path.join(file_path, "multinestresume.dat"), "w+") as resume:
        resume.write(
            " F\n"
            "        3000       12345           1          50\n"
            "    0.502352236277967168E+05    0.502900436569068333E+05\n"
            " T\n"
            "   0\n"
            " T F     0          50\n"
            "    0.648698272260014622E-26    0.502352236277967168E+05    0.502900436569068333E+05\n"
        )


class TestMulitNest:
    def test__loads_from_config_file_if_not_input(self):
        multinest = af.MultiNest(
            prior_passer=af.PriorPasser(sigma=2.0, use_errors=False, use_widths=False),
            n_live_points=40,
            sampling_efficiency=0.5,
            const_efficiency_mode=True,
            evidence_tolerance=0.4,
            importance_nested_sampling=False,
            multimodal=False,
            n_iter_before_update=90,
            null_log_evidence=-1.0e80,
            max_modes=50,
            mode_tolerance=-1e88,
            seed=0,
            verbose=True,
            resume=False,
            context=1,
            write_output=False,
            log_zero=-1e90,
            max_iter=1,
            init_MPI=True,
        )

        assert multinest.prior_passer.sigma == 2.0
        assert multinest.prior_passer.use_errors is False
        assert multinest.prior_passer.use_widths is False
        assert multinest.config_dict_search["n_live_points"] == 40
        assert multinest.config_dict_search["sampling_efficiency"] == 0.5

        multinest = af.MultiNest()

        assert multinest.prior_passer.sigma == 3.0
        assert multinest.prior_passer.use_errors is True
        assert multinest.prior_passer.use_widths is True
        assert multinest.config_dict_search["n_live_points"] == 50
        assert multinest.config_dict_search["sampling_efficiency"] == 0.6

        model = af.ModelMapper(mock_class_1=mock.MockClassx4)

        fitness = af.MultiNest.Fitness(
            analysis=None,
            model=model,
            samples_from_model=multinest.samples_from,
            stagger_resampling_likelihood=False,
            paths=None
        )

        assert fitness.model == model

    def test__read_quantities_from_weighted_samples_file(self, multinest_samples_path):
        multinest = af.MultiNest()
        multinest.paths = af.DirectoryPaths(path_prefix=path.join("non_linear", "multinest"))

        create_weighted_samples_4_parameters(file_path=multinest.paths.path)

        parameters = samp.parameters_from_file_weighted_samples(
            file_weighted_samples=path.join(multinest.paths.path, "multinest.txt"),
            prior_count=4,
        )

        assert parameters == [
            [1.1, 2.1, 3.1, 4.1],
            [0.9, 1.9, 2.9, 3.9],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        ]

        log_likelihood_list = samp.log_likelihood_list_from_file_weighted_samples(
            file_weighted_samples=path.join(multinest.paths.path, "multinest.txt")
        )

        value = -0.5 * 9999999.9

        assert log_likelihood_list == 10 * [value]

        weight_list = samp.weight_list_from_file_weighted_samples(
            file_weighted_samples=path.join(multinest.paths.path, "multinest.txt")
        )

        assert weight_list == [0.02, 0.02, 0.01, 0.05, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3]

    def test__read_total_samples_from_file_resume(self, multinest_resume_path):
        multinest = af.MultiNest()

        create_resume(file_path=multinest.paths.path)

        total_samples =samp.total_samples_from_file_resume(
            file_resume=path.join(multinest.paths.path, "multinestresume.dat")
        )

        assert total_samples == 12345

    def test__log_evidence_from_file_summary(self, multinest_summary_path):
        multinest = af.MultiNest()
        multinest.paths = af.DirectoryPaths(path_prefix=path.join("non_linear", "multinest"))

        create_summary_4_parameters(file_path=multinest.paths.samples_path)

        log_evidence = samp.log_evidence_from_file_summary(
            file_summary=path.join(multinest.paths.samples_path, "multinestsummary.txt"),
            prior_count=4,
        )

        assert log_evidence == 0.02

    def test__samples_from_model(
            self, multinest_samples_path, multinest_resume_path, multinest_summary_path
    ):
        multinest = af.MultiNest()
        multinest.paths = af.DirectoryPaths(path_prefix=path.join("non_linear", "multinest"))

        create_weighted_samples_4_parameters(file_path=multinest.paths.samples_path)
        create_resume(file_path=multinest.paths.samples_path)
        create_summary_4_parameters(file_path=multinest.paths.samples_path)

        model = af.ModelMapper(mock_class=mock.MockClassx4)
        model.mock_class.two = af.LogUniformPrior(lower_limit=1e-8, upper_limit=10.0)

        samples = multinest.samples_from(model=model)

        assert samples.parameter_lists == [
            [1.1, 2.1, 3.1, 4.1],
            [0.9, 1.9, 2.9, 3.9],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        ]

        value = -0.5 * 9999999.9

        assert samples.log_likelihood_list == 10 * [value]
        assert samples.log_prior_list == pytest.approx(
            [0.243902, 0.256410, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25], 1.0e-4
        )
        assert samples.weight_list == [0.02, 0.02, 0.01, 0.05, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3]
        assert samples.total_samples == 12345
        assert samples.log_evidence == 0.02
        assert samples.number_live_points == 50
