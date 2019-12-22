import os
import shutil
from functools import wraps

import pytest

import autofit as af
from autofit import Paths
from autofit.optimize.non_linear.output import AbstractOutput
from test_autofit.mock import MockClassNLOx4, MockClassNLOx5, MockClassNLOx6

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


class MockOutput(AbstractOutput):

    def __init__(self, model,
                 paths,
                 most_probable_model_parameters=None,
                 most_likely_model_parameters=None,
                 model_parameters_at_sigma_limit=None):

        super(MockOutput, self).__init__(model=model, paths=paths)

        self._most_probable_model_parameters = most_probable_model_parameters
        self._most_likely_model_parameters = most_likely_model_parameters
        self._model_parameters_at_sigma_limit = model_parameters_at_sigma_limit

    @property
    def most_probable_model_parameters(self):
        return self._most_probable_model_parameters

    @property
    def most_likely_model_parameters(self):
        return self._most_likely_model_parameters

    def model_parameters_at_sigma_limit(self, sigma_limit):
        return [(sigma_limit * value[0], sigma_limit * value[1]) for value in self._model_parameters_at_sigma_limit]


class TestOutput:
    def test__most_probable_model_instance(self):

        model = af.ModelMapper(
            mock_class_1=MockClassNLOx4, mock_class_2=MockClassNLOx6
        )

        output = MockOutput(model=model, paths=Paths(),
                        most_probable_model_parameters=[1.0, 2.0, 3.0, 4.0, -5.0, -6.0, -7.0, -8.0, 9.0, 10.0])

        most_probable = output.most_probable_model_instance

        assert most_probable.mock_class_1.one == 1.0
        assert most_probable.mock_class_1.two == 2.0
        assert most_probable.mock_class_1.three == 3.0
        assert most_probable.mock_class_1.four == 4.0

        assert most_probable.mock_class_2.one == (-5.0, -6.0)
        assert most_probable.mock_class_2.two == (-7.0, -8.0)
        assert most_probable.mock_class_2.three == 9.0
        assert most_probable.mock_class_2.four == 10.0

    def test__most_likely_model_instance(self):

        model = af.ModelMapper(
            mock_class_1=MockClassNLOx4, mock_class_2=MockClassNLOx6
        )

        output = MockOutput(model=model, paths=Paths(),
                        most_likely_model_parameters=[
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
        ])
        
        most_likely = output.most_likely_model_instance

        assert most_likely.mock_class_1.one == 21.0
        assert most_likely.mock_class_1.two == 22.0
        assert most_likely.mock_class_1.three == 23.0
        assert most_likely.mock_class_1.four == 24.0

        assert most_likely.mock_class_2.one == (25.0, -26.0)
        assert most_likely.mock_class_2.two == (-27.0, 28.0)
        assert most_likely.mock_class_2.three == 29.0
        assert most_likely.mock_class_2.four == 30.0

    def test__model_parameters_at_upper_and_lower_sigma_limit(
        self,
    ):

        model = af.ModelMapper(mock_class=MockClassNLOx4)
        output = MockOutput(
            model=model, paths=Paths(),
            most_probable_model_parameters=[1.0, 2.0, 3.0, 4.1],
            model_parameters_at_sigma_limit=[(0.88, 1.12), (1.88, 2.12), (2.88, 3.12), (3.88, 4.12)])

        params_upper = output.model_parameters_at_upper_sigma_limit(sigma_limit=1.0)
        assert params_upper == pytest.approx([1.12, 2.12, 3.12, 4.12], 1e-2)
        params_lower = output.model_parameters_at_lower_sigma_limit(sigma_limit=1.0)
        assert params_lower == pytest.approx([0.88, 1.88, 2.88, 3.88], 1e-2)

        params_upper = output.model_parameters_at_upper_sigma_limit(sigma_limit=2.0)
        assert params_upper == pytest.approx([2.0*1.12, 2.0*2.12, 2.0*3.12, 2.0*4.12], 1e-2)
        params_lower = output.model_parameters_at_lower_sigma_limit(sigma_limit=2.0)
        assert params_lower == pytest.approx([2.0*0.88, 2.0*1.88, 2.0*2.88, 2.0*3.88], 1e-2)

    def test__gaussian_priors(self):

        model = af.ModelMapper(mock_class=MockClassNLOx4)
        output = MockOutput(model=model, paths=Paths(),
                            most_probable_model_parameters=[1.0, 2.0, 3.0, 4.1],
                            model_parameters_at_sigma_limit=[(0.88, 1.12), (1.88, 2.12), (2.88, 3.12), (3.88, 4.12)])

        gaussian_priors = output.gaussian_priors_at_sigma_limit(sigma_limit=1.0)

        assert gaussian_priors[0][0] == 1.0
        assert gaussian_priors[1][0] == 2.0
        assert gaussian_priors[2][0] == 3.0
        assert gaussian_priors[3][0] == 4.1

        assert gaussian_priors[0][1] == pytest.approx(0.12, 1e-2)
        assert gaussian_priors[1][1] == pytest.approx(0.12, 1e-2)
        assert gaussian_priors[2][1] == pytest.approx(0.12, 1e-2)
        assert gaussian_priors[3][1] == pytest.approx(0.22, 1e-2)

    def test__offset_from_input(self, mn_summary_path):
        af.conf.instance.output_path = mn_summary_path + "/1_class"

        model = af.ModelMapper(mock_class=MockClassNLOx4)
        output = AbstractOutput(model, Paths())
        create_summary_4_parameters(path=output.paths.backup_path)

        # output.most_probable_model_parameters == [1.0, -2.0, 3.0, 4.0]

        offset_values = output.offset_values_from_input_model_parameters(
            input_model_parameters=[1.0, 1.0, 2.0, 3.0]
        )

        assert offset_values == [0.0, -3.0, 1.0, 1.0]

        af.conf.instance.output_path = mn_summary_path + "/2_classes"

        model = af.ModelMapper(
            mock_class_1=MockClassNLOx4, mock_class_2=MockClassNLOx6
        )
        output = AbstractOutput(model, Paths())
        create_summary_10_parameters(path=output.paths.backup_path)

        # output.most_probable_model_parameters == [1.0, 2.0, 3.0, 4.0, -5.0, -6.0, -7.0, -8.0, 9.0, 10.0]

        offset_values = output.offset_values_from_input_model_parameters(
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

        model = af.ModelMapper(mock_class=MockClassNLOx4)
        output = AbstractOutput(model, Paths())
        create_weighted_samples_4_parameters(path=output.paths.backup_path)

        model = output.sample_model_parameters_from_sample_index(sample_index=0)
        instance = output.sample_model_instance_from_sample_index(sample_index=0)
        weight = output.sample_weight_from_sample_index(sample_index=0)
        likelihood = output.sample_likelihood_from_sample_index(sample_index=0)

        assert output.total_samples == 10
        assert model == [1.1, 2.1, 3.1, 4.1]
        assert instance.mock_class.one == 1.1
        assert instance.mock_class.two == 2.1
        assert instance.mock_class.three == 3.1
        assert instance.mock_class.four == 4.1
        assert weight == 0.02
        assert likelihood == -0.5 * 9999999.9

        model = output.sample_model_parameters_from_sample_index(sample_index=5)
        instance = output.sample_model_instance_from_sample_index(sample_index=5)
        weight = output.sample_weight_from_sample_index(sample_index=5)
        likelihood = output.sample_likelihood_from_sample_index(sample_index=5)

        assert output.total_samples == 10
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

        model = af.ModelMapper(
            mock_class_1=MockClassNLOx4, mock_class_2=MockClassNLOx6
        )
        output = AbstractOutput(model, Paths())
        create_weighted_samples_10_parameters(path=output.paths.backup_path)

        model = output.sample_model_parameters_from_sample_index(sample_index=0)
        instance = output.sample_model_instance_from_sample_index(sample_index=0)
        weight = output.sample_weight_from_sample_index(sample_index=0)
        likelihood = output.sample_likelihood_from_sample_index(sample_index=0)

        assert output.total_samples == 10
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

        model = output.sample_model_parameters_from_sample_index(sample_index=5)
        instance = output.sample_model_instance_from_sample_index(sample_index=5)
        weight = output.sample_weight_from_sample_index(sample_index=5)
        likelihood = output.sample_likelihood_from_sample_index(sample_index=5)

        assert output.total_samples == 10
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

    def test__1_species__errors_1d_vectors_via_weighted_samples__1d_vectors_are_correct(
        self, mn_samples_path
    ):
        af.conf.instance.output_path = mn_samples_path + "/1_class"

        model = af.ModelMapper(mock_class=MockClassNLOx4)
        output = AbstractOutput(model, Paths())
        create_weighted_samples_4_parameters(path=output.paths.backup_path)

        model_errors = output.model_errors_at_sigma_limit(sigma_limit=3.0)
        assert model_errors == pytest.approx(
            [1.12 - 0.88, 2.12 - 1.88, 3.12 - 2.88, 4.12 - 3.88], 1e-2
        )

        model_errors_instance = output.model_errors_instance_at_sigma_limit(sigma_limit=3.0)
        assert model_errors_instance.mock_class.one == pytest.approx(1.12 - 0.88, 1e-2)
        assert model_errors_instance.mock_class.two == pytest.approx(2.12 - 1.88, 1e-2)
        assert model_errors_instance.mock_class.three == pytest.approx(
            3.12 - 2.88, 1e-2
        )
        assert model_errors_instance.mock_class.four == pytest.approx(4.12 - 3.88, 1e-2)

    def test__1_species__change_limit_to_1_sigma(self, mn_samples_path):
        af.conf.instance.output_path = mn_samples_path + "/1_class"

        model = af.ModelMapper(mock_class=MockClassNLOx4)
        output = AbstractOutput(model, Paths())
        create_weighted_samples_4_parameters(path=output.paths.backup_path)

        model_errors = output.model_errors_at_sigma_limit(sigma_limit=1.0)
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
