import os

import pytest

from autofit import Paths
from autoconf import conf
import autofit as af
from autofit.optimize.non_linear.nested_sampling.dynesty import DynestyOutput
from test_autofit.mock import MockClassNLOx4

directory = os.path.dirname(os.path.realpath(__file__))
pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        config_path=os.path.join(directory, "files/dynesty/config"),
        output_path=os.path.join(directory, "files/dynesty/output")
    )


@pytest.fixture(name="dynesty_output_converged")
def test_dynesty_output():
    dynesty_output_path = "{}/files/dynesty/".format(
        os.path.dirname(os.path.realpath(__file__))
    )

    af.conf.instance.output_path = dynesty_output_path

    mapper = af.ModelMapper(mock_class_1=MockClassNLOx4)

    return DynestyOutput(
        mapper,
        Paths(),
    )


class TestDynestyConfig:

    def test__loads_from_config_file_correct(self):

        dynesty = af.Dynesty()

        assert dynesty.iterations_per_update == 500
        assert dynesty.terminate_at_acceptance_ratio == True
        assert dynesty.acceptance_ratio_threshold == 2.0


class TestDynestyOutputConverged:
    def test__maximum_log_likelihood_and_evidence__from_summary(
        self, dynesty_output_converged
    ):

        assert dynesty_output_converged.maximum_log_likelihood == pytest.approx(
            9999999.9, 1.0e-4
        )
        assert dynesty_output_converged.evidence == pytest.approx(0.02, 1.0e-4)

    def test__most_probable_vector__from_summary(self, dynesty_summary_path):

        af.conf.instance.output_path = dynesty_summary_path + "/2_classes"

        model = af.ModelMapper(mock_class_1=MockClassNLOx4, mock_class_2=MockClassNLOx6)
        multinest_output = DynestyOutput(model, Paths())

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

    def test__most_likely_vector__from_summary(self, dynesty_summary_path):
        af.conf.instance.output_path = dynesty_summary_path + "/2_classes"

        model = af.ModelMapper(mock_class_1=MockClassNLOx4, mock_class_2=MockClassNLOx6)
        multinest_output = DynestyOutput(model, Paths())

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

    def test__vector_at_sigma__from_weighted_samples(self, dynesty_samples_path):
        af.conf.instance.output_path = dynesty_samples_path + "/1_class"

        model = af.ModelMapper(mock_class=MockClassNLOx4)
        multinest_output = DynestyOutput(model, Paths())
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
        self, dynesty_samples_path
    ):
        af.conf.instance.output_path = dynesty_samples_path + "/1_class"

        model = af.ModelMapper(mock_class=MockClassNLOx4)
        multinest_output = DynestyOutput(model, Paths())
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

    def test__total_samples__accepted_samples__acceptance_ratio__from_resume_file(self, dynesty_resume_path):

        af.conf.instance.output_path = dynesty_resume_path + "/2_classes"

        model = af.ModelMapper(mock_class_1=MockClassNLOx4, mock_class_2=MockClassNLOx6)
        multinest_output = DynestyOutput(model, Paths())

        create_resume(path=multinest_output.paths.backup_path)

        assert multinest_output.accepted_samples == 3000
        assert multinest_output.total_samples == 12345
        assert multinest_output.acceptance_ratio == 3000 / 12345

        create_resume_2(path=multinest_output.paths.backup_path)

        assert multinest_output.accepted_samples == 60
        assert multinest_output.total_samples == 60
        assert multinest_output.acceptance_ratio == 1.0


class TestDynestyOutputUnconverged:
    def test__maximum_log_likelihood_and_evidence__from_phys_live_points(
        self, dynesty_phys_live_path
    ):

        af.conf.instance.output_path = dynesty_phys_live_path + "/1_class"

        model = af.ModelMapper(mock_class_1=MockClassNLOx4)
        multinest_output = DynestyOutput(model, Paths())

        create_phys_live_4_parameters(path=multinest_output.paths.backup_path)

        assert multinest_output.maximum_log_likelihood == pytest.approx(0.04, 1.0e-4)
        assert multinest_output.evidence == None

    def test__most_probable_vector__use_most_likely_if_no_summary(
        self, dynesty_phys_live_path
    ):
        af.conf.instance.output_path = dynesty_phys_live_path + "/1_class"

        model = af.ModelMapper(mock_class_1=MockClassNLOx4)
        multinest_output = DynestyOutput(model, Paths())

        create_phys_live_4_parameters(path=multinest_output.paths.backup_path)

        assert multinest_output.most_probable_vector == [1.0, 2.0, 3.0, 5.0]

    def test__most_likely_vector__from_phys_live_points(
        self, dynesty_phys_live_path
    ):
        af.conf.instance.output_path = dynesty_phys_live_path + "/1_class"

        model = af.ModelMapper(mock_class_1=MockClassNLOx4)
        multinest_output = DynestyOutput(model, Paths())

        create_phys_live_4_parameters(path=multinest_output.paths.backup_path)

        assert multinest_output.most_likely_vector == [1.0, 2.0, 3.0, 5.0]

    def test__vector_at_sigma__uses_min_max_of_phys_lives(
        self, dynesty_phys_live_path
    ):
        af.conf.instance.output_path = dynesty_phys_live_path + "/1_class"

        model = af.ModelMapper(mock_class=MockClassNLOx4)
        multinest_output = DynestyOutput(model, Paths())
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


class TestCopyWithNameExtension:
    @staticmethod
    def assert_non_linear_attributes_equal(copy):
        assert copy.paths.phase_name == "phase_name/one"

    def test_dynesty(self):
        optimizer = af.Dynesty(Paths("phase_name"), sigma=2.0)

        copy = optimizer.copy_with_name_extension("one")
        self.assert_non_linear_attributes_equal(copy)
        assert isinstance(copy, af.Dynesty)
        assert copy.sigma is optimizer.sigma
        assert copy.iterations_per_update is optimizer.iterations_per_update
        assert copy.terminate_at_acceptance_ratio is optimizer.terminate_at_acceptance_ratio
        assert copy.acceptance_ratio_threshold is optimizer.acceptance_ratio_threshold
