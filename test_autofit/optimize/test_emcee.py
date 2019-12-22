import os
import shutil
from functools import wraps

import pytest

import autofit as af
from autofit import Paths
from autofit.optimize.non_linear.output import EmceeOutput
from test_autofit.mock import MockClassNLOx4, MockClassNLOx5, MockClassNLOx6

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(scope="session", autouse=True)
def do_something():
    af.conf.instance = af.conf.Config(
        "{}/../test_files/configs/non_linear".format(
            os.path.dirname(os.path.realpath(__file__))
        )
    )


@pytest.fixture(name="emcee_output")
def test_emcee_output():
    emcee_output_path = "{}/../test_files/non_linear/emcee/".format(
        os.path.dirname(os.path.realpath(__file__))
    )

    af.conf.instance.output_path = emcee_output_path

    mapper = af.ModelMapper(
        mock_class_1=MockClassNLOx4,
    )

    return EmceeOutput(mapper, Paths())


class TestEmceeOutput:

    def test__maximum_log_likelihood(self, emcee_output):

        assert emcee_output.maximum_log_likelihood == pytest.approx(-60560.20617, 1.0e-4)

    def test__most_probable_parameters(self, mn_summary_path):
        af.conf.instance.output_path = mn_summary_path + "/2_classes"

        mapper = af.ModelMapper(
            mock_class_1=MockClassNLOx4,
        )
        emcee_output = EmceeOutput(mapper, Paths())

        assert emcee_output.most_probable_model_parameters == [
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

    def test__most_likely_parameters(self, emcee_output):

        assert emcee_output.most_likely_model_parameters == pytest.approx([
            -0.06985,
        0.04442,
         6.68523,
            1.11980,
        ], 1.0e-3)

class TestSamples(object):
    def test__1_class___model_parameters_instance_weight_and_likelihood(
        self, mn_samples_path
    ):
        af.conf.instance.output_path = mn_samples_path + "/1_class"

        mapper = af.ModelMapper(mock_class=MockClassNLOx4)
        emcee_output = EmceeOutput(mapper, Paths())
        create_weighted_samples_4_parameters(path=emcee_output.paths.backup_path)

        model = emcee_output.sample_model_parameters_from_sample_index(sample_index=0)
        instance = emcee_output.sample_model_instance_from_sample_index(sample_index=0)
        weight = emcee_output.sample_weight_from_sample_index(sample_index=0)
        likelihood = emcee_output.sample_likelihood_from_sample_index(sample_index=0)

        assert emcee_output.total_samples == 10
        assert model == [1.1, 2.1, 3.1, 4.1]
        assert instance.mock_class.one == 1.1
        assert instance.mock_class.two == 2.1
        assert instance.mock_class.three == 3.1
        assert instance.mock_class.four == 4.1
        assert weight == 0.02
        assert likelihood == -0.5 * 9999999.9

        model = emcee_output.sample_model_parameters_from_sample_index(sample_index=5)
        instance = emcee_output.sample_model_instance_from_sample_index(sample_index=5)
        weight = emcee_output.sample_weight_from_sample_index(sample_index=5)
        likelihood = emcee_output.sample_likelihood_from_sample_index(sample_index=5)

        assert emcee_output.total_samples == 10
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
        emcee_output = EmceeOutput(mapper, Paths())
        create_weighted_samples_10_parameters(path=emcee_output.paths.backup_path)

        model = emcee_output.sample_model_parameters_from_sample_index(sample_index=0)
        instance = emcee_output.sample_model_instance_from_sample_index(sample_index=0)
        weight = emcee_output.sample_weight_from_sample_index(sample_index=0)
        likelihood = emcee_output.sample_likelihood_from_sample_index(sample_index=0)

        assert emcee_output.total_samples == 10
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

        model = emcee_output.sample_model_parameters_from_sample_index(sample_index=5)
        instance = emcee_output.sample_model_instance_from_sample_index(sample_index=5)
        weight = emcee_output.sample_weight_from_sample_index(sample_index=5)
        likelihood = emcee_output.sample_likelihood_from_sample_index(sample_index=5)

        assert emcee_output.total_samples == 10
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
        emcee_output = EmceeOutput(mapper, Paths())
        create_weighted_samples_4_parameters(path=emcee_output.paths.backup_path)

        params_upper = emcee_output.model_parameters_at_upper_sigma_limit(sigma_limit=3.0)
        assert params_upper == pytest.approx([1.12, 2.12, 3.12, 4.12], 1e-2)
        params_lower = emcee_output.model_parameters_at_lower_sigma_limit(sigma_limit=3.0)
        assert params_lower == pytest.approx([0.88, 1.88, 2.88, 3.88], 1e-2)

    def test__1_profile__change_limit_to_1_sigma(self, mn_samples_path):
        af.conf.instance.output_path = mn_samples_path + "/1_class"

        mapper = af.ModelMapper(mock_class=MockClassNLOx4)
        emcee_output = EmceeOutput(mapper, Paths())
        create_weighted_samples_4_parameters(path=emcee_output.paths.backup_path)

        params_upper = emcee_output.model_parameters_at_upper_sigma_limit(sigma_limit=1.0)
        assert params_upper == pytest.approx([1.07, 2.07, 3.07, 4.07], 1e-2)
        params_lower = emcee_output.model_parameters_at_lower_sigma_limit(sigma_limit=1.0)
        assert params_lower == pytest.approx([0.93, 1.93, 2.93, 3.93], 1e-2)

    def test__1_species__errors_1d_vectors_via_weighted_samples__1d_vectors_are_correct(
        self, mn_samples_path
    ):
        af.conf.instance.output_path = mn_samples_path + "/1_class"

        mapper = af.ModelMapper(mock_class=MockClassNLOx4)
        emcee_output = EmceeOutput(mapper, Paths())
        create_weighted_samples_4_parameters(path=emcee_output.paths.backup_path)

        model_errors = emcee_output.model_errors_at_sigma_limit(sigma_limit=3.0)
        assert model_errors == pytest.approx(
            [1.12 - 0.88, 2.12 - 1.88, 3.12 - 2.88, 4.12 - 3.88], 1e-2
        )

        model_errors_instance = emcee_output.model_errors_instance_at_sigma_limit(sigma_limit=3.0)
        assert model_errors_instance.mock_class.one == pytest.approx(1.12 - 0.88, 1e-2)
        assert model_errors_instance.mock_class.two == pytest.approx(2.12 - 1.88, 1e-2)
        assert model_errors_instance.mock_class.three == pytest.approx(
            3.12 - 2.88, 1e-2
        )
        assert model_errors_instance.mock_class.four == pytest.approx(4.12 - 3.88, 1e-2)

    def test__1_species__change_limit_to_1_sigma(self, mn_samples_path):
        af.conf.instance.output_path = mn_samples_path + "/1_class"

        mapper = af.ModelMapper(mock_class=MockClassNLOx4)
        emcee_output = EmceeOutput(mapper, Paths())
        create_weighted_samples_4_parameters(path=emcee_output.paths.backup_path)

        model_errors = emcee_output.model_errors_at_sigma_limit(sigma_limit=1.0)
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

    multi_nest = af.MultiNest(run=run, paths=Paths(phase_name="", remove_files=False))

    create_weighted_samples_4_parameters(multi_nest.paths.sym_path)
    create_summary_4_parameters(multi_nest.paths.sym_path)

    return multi_nest


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
