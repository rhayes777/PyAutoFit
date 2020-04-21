import os

import pytest

from autofit import Paths
from autoconf import conf
import autofit as af
import numpy as np
import pickle
from test_autofit.mock import MockClassNLOx4

directory = os.path.dirname(os.path.realpath(__file__))
pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        config_path=os.path.join(directory, "files/dynesty/config"),
        output_path=os.path.join(directory, "files/dynesty/output"),
    )


class MockDynestyResults:

    def __init__(self, samples, logl, logwt, ncall, logz, nlive):
        self.samples = samples
        self.logl = logl
        self.logwt = logwt
        self.ncall = ncall
        self.logz = logz
        self.nlive = nlive


class MockDynestySampler:

    def __init__(self, results):
        self.results = results


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

    def test__samples_from_model(self):
        # Setup pickle of mock Dynesty sampler that the samples_from_model function uses.

        results = MockDynestyResults(samples=[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
                                     logl=[1.0, 2.0, 3.0], logwt=[1.0, 2.0, 3.0], ncall=[5.0, 5.0],
                                     logz=[10.0, 11.0, 12.0], nlive=3)

        sampler = MockDynestySampler(results=results)

        paths = af.Paths()

        with open(
                f"{paths.chains_path}/dynesty.pickle", "wb"
        ) as f:
            pickle.dump(sampler, f)

        dynesty = af.DynestyStatic(paths=paths)

        model = af.ModelMapper(mock_class=MockClassNLOx4)
        model.mock_class.two = af.LogUniformPrior(lower_limit=0.0, upper_limit=10.0)

        samples = dynesty.samples_from_model(model=model, paths=paths)

        assert samples.parameters == [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]
        assert samples.log_likelihoods == [1.0, 2.0, 3.0]
        assert samples.log_priors == [0.25, 0.25, 0.25]
        assert samples.weights == [1.0, 2.0, 3.0]
        assert samples.total_samples == 10
        assert samples.log_evidence == 12.0
        assert samples.number_live_points == 3


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
