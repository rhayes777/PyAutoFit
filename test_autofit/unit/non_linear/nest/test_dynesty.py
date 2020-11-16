from os import path
import pickle
import sys

import numpy as np
import pytest

import autofit as af
from autoconf import conf
from autofit.mock import mock

directory = path.dirname(path.realpath(__file__))
pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance.push(
        new_path=path.join(directory, "files", "dynesty", "config"),
        output_path=path.join(directory, "files", "dynesty", "output"),
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
    def test__loads_from_config_file_if_not_input(self):
        dynesty = af.DynestyStatic(
            prior_passer=af.PriorPasser(sigma=2.0, use_errors=False, use_widths=False),
            n_live_points=151,
            facc=0.6,
            evidence_tolerance=0.1,
            bound="ellipse",
            sample="manual",
            update_interval=True,
            bootstrap=1.0,
            enlarge=2.0,
            vol_dec=2.1,
            vol_check=2.2,
            walks=26,
            slices=6,
            fmove=0.8,
            max_move=101,
            maxiter=2,
            maxcall=3,
            logl_max=1.0,
            n_effective=4,
            iterations_per_update=501,
            terminate_at_acceptance_ratio=False,
            acceptance_ratio_threshold=0.5,
            number_of_cores=2,
        )

        assert dynesty.prior_passer.sigma == 2.0
        assert dynesty.prior_passer.use_errors == False
        assert dynesty.prior_passer.use_widths == False
        assert dynesty.iterations_per_update == 501
        assert dynesty.n_live_points == 151
        assert dynesty.facc == 0.6
        assert dynesty.evidence_tolerance == 0.1
        assert dynesty.bound == "ellipse"
        assert dynesty.sample == "manual"
        assert dynesty.update_interval == True
        assert dynesty.bootstrap == 1.0
        assert dynesty.enlarge == 2.0
        assert dynesty.vol_dec == 2.1
        assert dynesty.vol_check == 2.2
        assert dynesty.walks == 26
        assert dynesty.slices == 6
        assert dynesty.fmove == 0.8
        assert dynesty.max_move == 101
        assert dynesty.maxiter == 2
        assert dynesty.maxcall == 3
        assert dynesty.logl_max == 1.0
        assert dynesty.n_effective == 4
        assert dynesty.terminate_at_acceptance_ratio == False
        assert dynesty.acceptance_ratio_threshold == 0.5
        assert dynesty.number_of_cores == 2

        dynesty = af.DynestyStatic()

        assert dynesty.prior_passer.sigma == 3.0
        assert dynesty.prior_passer.use_errors == True
        assert dynesty.prior_passer.use_widths == True
        assert dynesty.iterations_per_update == 500
        assert dynesty.n_live_points == 150
        assert dynesty.evidence_tolerance == 0.159
        assert dynesty.bound == "multi"
        assert dynesty.sample == "auto"
        assert dynesty.update_interval == None
        assert dynesty.bootstrap == 0.0
        assert dynesty.enlarge == 1.0
        assert dynesty.vol_dec == 0.5
        assert dynesty.vol_check == 2.0
        assert dynesty.walks == 5
        assert dynesty.facc == 0.5
        assert dynesty.slices == 5
        assert dynesty.fmove == 0.9
        assert dynesty.max_move == 100
        assert dynesty.maxiter == sys.maxsize
        assert dynesty.maxcall == sys.maxsize
        assert dynesty.logl_max == np.inf
        assert dynesty.n_effective == np.inf
        assert dynesty.terminate_at_acceptance_ratio == True
        assert dynesty.acceptance_ratio_threshold == 2.0
        assert dynesty.number_of_cores == 1

        dynesty = af.DynestyDynamic(
            prior_passer=af.PriorPasser(sigma=2.0, use_errors=False, use_widths=False),
            n_live_points=0,
            iterations_per_update=501,
            facc=0.6,
            evidence_tolerance=0.2,
            bound="ellipse",
            sample="manual",
            update_interval=True,
            bootstrap=1.0,
            enlarge=2.0,
            vol_dec=2.1,
            vol_check=2.2,
            walks=26,
            slices=6,
            fmove=0.8,
            max_move=101,
            maxiter=2,
            maxcall=3,
            logl_max=1.0,
            n_effective=4,
            terminate_at_acceptance_ratio=False,
            acceptance_ratio_threshold=0.5,
            number_of_cores=3,
        )

        assert dynesty.prior_passer.sigma == 2.0
        assert dynesty.prior_passer.use_errors == False
        assert dynesty.prior_passer.use_widths == False
        assert dynesty.n_live_points == 500
        assert dynesty.iterations_per_update == 501
        assert dynesty.facc == 0.6
        assert dynesty.evidence_tolerance == 0.2
        assert dynesty.bound == "ellipse"
        assert dynesty.sample == "manual"
        assert dynesty.update_interval == True
        assert dynesty.bootstrap == 1.0
        assert dynesty.enlarge == 2.0
        assert dynesty.vol_dec == 2.1
        assert dynesty.vol_check == 2.2
        assert dynesty.walks == 26
        assert dynesty.slices == 6
        assert dynesty.fmove == 0.8
        assert dynesty.max_move == 101
        assert dynesty.maxiter == 2
        assert dynesty.maxcall == 3
        assert dynesty.logl_max == 1.0
        assert dynesty.n_effective == 4
        assert dynesty.terminate_at_acceptance_ratio == False
        assert dynesty.acceptance_ratio_threshold == 0.5
        assert dynesty.number_of_cores == 3

        dynesty = af.DynestyDynamic()

        assert dynesty.prior_passer.sigma == 3.0
        assert dynesty.prior_passer.use_errors == True
        assert dynesty.prior_passer.use_widths == True
        assert dynesty.n_live_points == 5
        assert dynesty.iterations_per_update == 501
        assert dynesty.facc == 0.6
        assert dynesty.bound == "balls"
        assert dynesty.sample == "rwalk"
        assert dynesty.update_interval == 2.0
        assert dynesty.bootstrap == 1.0
        assert dynesty.enlarge == 2.0
        assert dynesty.vol_dec == 0.4
        assert dynesty.vol_check == 3.0
        assert dynesty.walks == 26
        assert dynesty.slices == 6
        assert dynesty.fmove == 0.8
        assert dynesty.max_move == 101
        assert dynesty.maxiter == sys.maxsize
        assert dynesty.maxcall == sys.maxsize
        assert dynesty.logl_max == np.inf
        assert dynesty.n_effective == np.inf
        assert dynesty.terminate_at_acceptance_ratio == True
        assert dynesty.acceptance_ratio_threshold == 2.0
        assert dynesty.number_of_cores == 4

    def test__tag(self):
        dynesty = af.DynestyStatic(
            n_live_points=40, bound="none", sample="auto", enlarge=1.0
        )

        assert (
            dynesty.tag
            == "dynesty_static[nlive_40__bound_none__enlarge_1.0__sample_auto]"
        )

        dynesty = af.DynestyStatic(
            n_live_points=41,
            bound="multi",
            sample="unif",
            enlarge=1.0,
            vol_dec=2.0,
            vol_check=3.0,
        )

        assert (
            dynesty.tag
            == "dynesty_static[nlive_41__bound_multi_vol_dec_2.0_vol_check_3.0__enlarge_1.0__sample_unif]"
        )

        dynesty = af.DynestyStatic(
            n_live_points=43,
            bound="single",
            sample="rwalk",
            enlarge=1.0,
            walks=1,
            facc=0.5,
        )

        assert (
            dynesty.tag
            == "dynesty_static[nlive_43__bound_single__enlarge_1.0__sample_rwalk_walks_1_facc_0.5]"
        )

        dynesty = af.DynestyStatic(
            n_live_points=44,
            bound="balls",
            sample="hslice",
            enlarge=1.0,
            slices=1,
            max_move=2,
        )

        assert (
            dynesty.tag
            == "dynesty_static[nlive_44__bound_balls__enlarge_1.0__sample_hslice_slices_1_max_move_2]"
        )

        dynesty = af.DynestyStatic(
            n_live_points=44, bound="balls", sample="slice", enlarge=1.0, slices=1
        )

        assert (
            dynesty.tag
            == "dynesty_static[nlive_44__bound_balls__enlarge_1.0__sample_slice_slices_1]"
        )

        dynesty = af.DynestyStatic(
            n_live_points=44, bound="balls", sample="rslice", enlarge=1.0, slices=1
        )

        assert (
            dynesty.tag
            == "dynesty_static[nlive_44__bound_balls__enlarge_1.0__sample_rslice_slices_1]"
        )

        dynesty = af.DynestyDynamic(
            bound="multi", sample="unif", enlarge=1.0, vol_dec=2.0, vol_check=3.0
        )

        assert (
            dynesty.tag
            == "dynesty_dynamic[nlive_5__bound_multi_vol_dec_2.0_vol_check_3.0__enlarge_1.0__sample_unif]"
        )

        dynesty = af.DynestyDynamic(
            bound="balls", sample="rslice", enlarge=1.0, slices=1
        )

        assert (
            dynesty.tag
            == "dynesty_dynamic[nlive_5__bound_balls__enlarge_1.0__sample_rslice_slices_1]"
        )

    def test__samples_from_model(self):
        # Setup pickle of mock Dynesty sampler that the samples_from_model function uses.

        results = MockDynestyResults(
            samples=np.array(
                [[1.0, 2.0, 3.0, 5.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]
            ),
            logl=[1.0, 2.0, 3.0],
            logwt=[np.log(1.0), np.log(2.0), np.log(3.0)],
            ncall=[5.0, 5.0],
            logz=[-2.0, -1.0, 0.0],
            nlive=3,
        )

        sampler = MockDynestySampler(results=results)

        paths = af.Paths()

        dynesty = af.DynestyStatic(paths=paths)

        with open(path.join(dynesty.paths.samples_path, "dynesty.pickle"), "wb") as f:
            pickle.dump(sampler, f)

        model = af.ModelMapper(mock_class=mock.MockClassx4)
        model.mock_class.two = af.LogUniformPrior(lower_limit=0.0, upper_limit=10.0)

        samples = dynesty.samples_via_sampler_from_model(model=model)

        assert isinstance(samples.parameters, list)
        assert isinstance(samples.parameters[0], list)
        assert isinstance(samples.log_likelihoods, list)
        assert isinstance(samples.log_priors, list)
        assert isinstance(samples.log_posteriors, list)
        assert isinstance(samples.weights, list)

        assert samples.parameters == [
            [1.0, 2.0, 3.0, 5.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        ]
        assert samples.log_likelihoods == [1.0, 2.0, 3.0]
        assert samples.log_priors == [0.2, 0.25, 0.25]
        assert samples.weights == pytest.approx([1.0, 2.0, 3.0], 1.0e-4)
        assert samples.total_samples == 10
        assert samples.log_evidence == 0.0
        assert samples.number_live_points == 3


class TestCopyWithNameExtension:
    @staticmethod
    def assert_non_linear_attributes_equal(copy):
        assert copy.paths.name == path.join("name", "one")

    def test_dynesty(self):
        search = af.DynestyStatic(af.Paths("name"))

        copy = search.copy_with_name_extension("one")
        self.assert_non_linear_attributes_equal(copy)
        assert isinstance(copy, af.DynestyStatic)
        assert copy.prior_passer is search.prior_passer
        assert (
            copy.terminate_at_acceptance_ratio is search.terminate_at_acceptance_ratio
        )
        assert copy.acceptance_ratio_threshold is search.acceptance_ratio_threshold

        assert copy.iterations_per_update is search.iterations_per_update
        assert copy.evidence_tolerance == search.evidence_tolerance
        assert copy.n_live_points == search.n_live_points
        assert copy.bound == search.bound
        assert copy.sample == search.sample
        assert copy.update_interval == search.update_interval
        assert copy.bootstrap == search.bootstrap
        assert copy.enlarge == search.enlarge
        assert copy.vol_dec == search.vol_dec
        assert copy.vol_check == search.vol_check
        assert copy.walks == search.walks
        assert copy.facc == search.facc
        assert copy.slices == search.slices
        assert copy.fmove == search.fmove
        assert copy.max_move == search.max_move
        assert copy.number_of_cores == search.number_of_cores

        search = af.DynestyDynamic(af.Paths("name"))

        copy = search.copy_with_name_extension("one")
        self.assert_non_linear_attributes_equal(copy)
        assert isinstance(copy, af.DynestyDynamic)
        assert copy.prior_passer is search.prior_passer
        assert (
            copy.terminate_at_acceptance_ratio is search.terminate_at_acceptance_ratio
        )
        assert copy.acceptance_ratio_threshold is search.acceptance_ratio_threshold

        assert copy.evidence_tolerance == search.evidence_tolerance
        assert copy.iterations_per_update is search.iterations_per_update
        assert copy.bound == search.bound
        assert copy.sample == search.sample
        assert copy.update_interval == search.update_interval
        assert copy.bootstrap == search.bootstrap
        assert copy.enlarge == search.enlarge
        assert copy.vol_dec == search.vol_dec
        assert copy.vol_check == search.vol_check
        assert copy.walks == search.walks
        assert copy.facc == search.facc
        assert copy.slices == search.slices
        assert copy.fmove == search.fmove
        assert copy.max_move == search.max_move
        assert copy.number_of_cores == search.number_of_cores
