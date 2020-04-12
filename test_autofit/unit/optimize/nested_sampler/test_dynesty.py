import os

import pytest

from autofit import Paths
from autoconf import conf
import autofit as af
import numpy as np
from autofit.optimize.non_linear.nested_sampling.dynesty import DynestyOutput
from test_autofit.mock import MockClassNLOx4

directory = os.path.dirname(os.path.realpath(__file__))
pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        config_path=os.path.join(directory, "files/dynesty/config"),
        output_path=os.path.join(directory, "files/dynesty/output"),
    )


@pytest.fixture(name="dynesty_output_converged")
def test_dynesty_output_converged():
    dynesty_output_path = "{}/files/dynesty/".format(
        os.path.dirname(os.path.realpath(__file__))
    )

    af.conf.instance.output_path = dynesty_output_path

    mapper = af.ModelMapper(mock_class_1=MockClassNLOx4)

    return DynestyOutput(mapper, Paths())


@pytest.fixture(name="dynesty_output_unconverged")
def test_dynesty_output_unconverged():
    dynesty_output_path = "{}/files/dynesty_unconverged/".format(
        os.path.dirname(os.path.realpath(__file__))
    )

    af.conf.instance.output_path = dynesty_output_path

    mapper = af.ModelMapper(mock_class_1=MockClassNLOx4)

    return DynestyOutput(mapper, Paths())


class TestDynestyConfig:
    def test__loads_from_config_file_correct(self):

        dynesty = af.DynestyStatic()

        assert dynesty.iterations_per_update == 500
        assert dynesty.n_live_points == 150
        assert dynesty.bound == "multi"
        assert dynesty.sample == "auto"
        assert dynesty.update_interval == None
        assert dynesty.bootstrap == 0.0
        assert dynesty.enlarge == 1.0
        assert dynesty.vol_dec == 0.5
        assert dynesty.vol_check == 2.0
        assert dynesty.walks == 25
        assert dynesty.facc == 0.5
        assert dynesty.slices == 5
        assert dynesty.fmove == 0.9
        assert dynesty.max_move == 100
        assert dynesty.terminate_at_acceptance_ratio == True
        assert dynesty.acceptance_ratio_threshold == 2.0

        dynesty = af.DynestyDynamic()

        assert dynesty.iterations_per_update == 501
        assert dynesty.bound == "balls"
        assert dynesty.sample == "rwalk"
        assert dynesty.update_interval == 2.0
        assert dynesty.bootstrap == 1.0
        assert dynesty.enlarge == 2.0
        assert dynesty.vol_dec == 0.4
        assert dynesty.vol_check == 3.0
        assert dynesty.walks == 26
        assert dynesty.facc == 0.6
        assert dynesty.slices == 6
        assert dynesty.fmove == 0.8
        assert dynesty.max_move == 101
        assert dynesty.terminate_at_acceptance_ratio == False
        assert dynesty.acceptance_ratio_threshold == 3.0


class TestDynestyOutputConverged:
    def test__maximum_log_likelihood_and_evidence__from_summary(
        self, dynesty_output_converged
    ):

        assert dynesty_output_converged.maximum_log_likelihood == pytest.approx(
            618.65239, 1.0e-4
        )
        assert dynesty_output_converged.evidence == pytest.approx(592.45643, 1.0e-4)

    def test__most_probable_vector__from_chains(self, dynesty_output_converged):

        assert dynesty_output_converged.most_probable_vector == pytest.approx(
            [10.011122, 0.4996056, 0.00232913, 0.00101658], 1.0e-4
        )

    def test__most_likely_vector__from_summary(self, dynesty_output_converged):

        assert dynesty_output_converged.most_likely_vector == pytest.approx(
            [10.0221, 0.49940, 0.002401, 0.00133179], 1.0e-4
        )

    def test__vector_at_sigma__from_weighted_samples(self, dynesty_output_converged):

        params = dynesty_output_converged.vector_at_sigma(sigma=3.0)

        assert params[0][0:2] == pytest.approx((9.599, 10.387), 1e-2)
        assert params[1][0:2] == pytest.approx((0.485, 0.518), 1e-2)
        assert params[2][0:2] == pytest.approx((-0.0180, 0.0255), 1e-2)
        assert params[3][0:2] == pytest.approx((-0.0209, 0.0269), 1e-2)

        params = dynesty_output_converged.vector_at_sigma(sigma=1.0)

        assert params[0][0:2] == pytest.approx((9.971, 10.069), 1e-2)
        assert params[1][0:2] == pytest.approx((0.497, 0.5014), 1e-2)
        assert params[2][0:2] == pytest.approx((-0.000225, 0.00515), 1e-2)
        assert params[3][0:2] == pytest.approx((-0.00204, 0.0042), 1e-2)

    def test__samples__model_parameters_weight_and_likelihood_from_sample_index(
        self, dynesty_output_converged
    ):

        # 629 is the most-likely model.

        model = dynesty_output_converged.vector_from_sample_index(sample_index=100)
        weight = dynesty_output_converged.weight_from_sample_index(sample_index=100)
        likelihood = dynesty_output_converged.likelihood_from_sample_index(
            sample_index=100
        )

        assert model == pytest.approx([11.84474, 5.13973, -0.045362, -0.093680], 1.0e-3)

        assert weight == pytest.approx(-85780.18, 1.0e-2)
        assert likelihood == pytest.approx(-85771.5, 1.0e-2)

        model = dynesty_output_converged.vector_from_sample_index(sample_index=629)
        weight = dynesty_output_converged.weight_from_sample_index(sample_index=629)
        likelihood = dynesty_output_converged.likelihood_from_sample_index(
            sample_index=629
        )

        assert model == pytest.approx([10.0221, 0.49940, 0.002401, 0.00133179], 1.0e-3)

        assert weight == pytest.approx(585.809, 1.0e-2)
        assert likelihood == pytest.approx(618.65, 1.0e-2)

    def test__total_samples__accepted_samples__acceptance_ratio(
        self, dynesty_output_converged
    ):

        assert dynesty_output_converged.total_accepted_samples == 610
        assert dynesty_output_converged.total_samples == 2282


class TestDynestyOutputUnconverged:
    def test__most_probable_vector__from_chains(self, dynesty_output_unconverged):

        assert dynesty_output_unconverged.most_probable_vector == pytest.approx(
            [571.815, 10.8658, -0.0048052, 0.0257509], 1.0e-2
        )

    def test__vector_at_sigma__from_weighted_samples(self, dynesty_output_unconverged):

        params = dynesty_output_unconverged.vector_at_sigma(sigma=3.0)

        assert params[0][0:2] == pytest.approx((1.98047, 66.3346), 1e-2)
        assert params[1][0:2] == pytest.approx((0.64778, 21.3195), 1e-2)
        assert params[2][0:2] == pytest.approx((-0.21020, 0.19028), 1e-2)
        assert params[3][0:2] == pytest.approx((-0.215614, 0.14256), 1e-2)

        params = dynesty_output_unconverged.vector_at_sigma(sigma=1.0)

        assert params[0][0:2] == pytest.approx((1.98047, 66.3346), 1e-2)
        assert params[1][0:2] == pytest.approx((0.64778, 21.3195), 1e-2)
        assert params[2][0:2] == pytest.approx((-0.21020, 0.19028), 1e-2)
        assert params[3][0:2] == pytest.approx((-0.215614, 0.14256), 1e-2)


class TestCopyWithNameExtension:
    @staticmethod
    def assert_non_linear_attributes_equal(copy):
        assert copy.paths.phase_name == "phase_name/one"

    def test_dynesty(self):
        optimizer = af.DynestyStatic(Paths("phase_name"), sigma=2.0)

        copy = optimizer.copy_with_name_extension("one")
        self.assert_non_linear_attributes_equal(copy)
        assert isinstance(copy, af.DynestyStatic)
        assert copy.sigma is optimizer.sigma
        assert (
            copy.terminate_at_acceptance_ratio
            is optimizer.terminate_at_acceptance_ratio
        )
        assert copy.acceptance_ratio_threshold is optimizer.acceptance_ratio_threshold

        assert copy.iterations_per_update is optimizer.iterations_per_update
        assert copy.n_live_points == optimizer.n_live_points
        assert copy.bound == optimizer.bound
        assert copy.sample == optimizer.sample
        assert copy.update_interval == optimizer.update_interval
        assert copy.bootstrap == optimizer.bootstrap
        assert copy.enlarge == optimizer.enlarge
        assert copy.vol_dec == optimizer.vol_dec
        assert copy.vol_check == optimizer.vol_check
        assert copy.walks == optimizer.walks
        assert copy.facc == optimizer.facc
        assert copy.slices == optimizer.slices
        assert copy.fmove == optimizer.fmove
        assert copy.max_move == optimizer.max_move

        optimizer = af.DynestyDynamic(Paths("phase_name"), sigma=2.0)

        copy = optimizer.copy_with_name_extension("one")
        self.assert_non_linear_attributes_equal(copy)
        assert isinstance(copy, af.DynestyDynamic)
        assert copy.sigma is optimizer.sigma
        assert (
            copy.terminate_at_acceptance_ratio
            is optimizer.terminate_at_acceptance_ratio
        )
        assert copy.acceptance_ratio_threshold is optimizer.acceptance_ratio_threshold

        assert copy.iterations_per_update is optimizer.iterations_per_update
        assert copy.bound == optimizer.bound
        assert copy.sample == optimizer.sample
        assert copy.update_interval == optimizer.update_interval
        assert copy.bootstrap == optimizer.bootstrap
        assert copy.enlarge == optimizer.enlarge
        assert copy.vol_dec == optimizer.vol_dec
        assert copy.vol_check == optimizer.vol_check
        assert copy.walks == optimizer.walks
        assert copy.facc == optimizer.facc
        assert copy.slices == optimizer.slices
        assert copy.fmove == optimizer.fmove
        assert copy.max_move == optimizer.max_move
