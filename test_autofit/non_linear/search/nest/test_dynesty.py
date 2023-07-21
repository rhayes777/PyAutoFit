from os import path

import numpy as np
import pytest

import autofit as af

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

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
            nlive=151,
            dlogz=0.1,
            iterations_per_update=501,
            number_of_cores=2,
        )

        assert dynesty.iterations_per_update == 501

        assert dynesty.config_dict_search["nlive"] == 151
        assert dynesty.config_dict_run["dlogz"] == 0.1
        assert dynesty.number_of_cores == 2

        dynesty = af.DynestyStatic()

        assert dynesty.iterations_per_update == 500

        assert dynesty.config_dict_search["nlive"] == 150
        assert dynesty.config_dict_run["dlogz"] == None
        assert dynesty.number_of_cores == 1

        dynesty = af.DynestyDynamic(
            facc=0.4,
            iterations_per_update=501,
            dlogz_init=0.2,
            number_of_cores=3,
        )

        assert dynesty.iterations_per_update == 501

        assert dynesty.config_dict_search["facc"] == 0.4
        assert dynesty.config_dict_run["dlogz_init"] == 0.2
        assert dynesty.number_of_cores == 3

        dynesty = af.DynestyDynamic()

        assert dynesty.iterations_per_update == 501

        assert dynesty.config_dict_search["facc"] == 0.6
        assert dynesty.config_dict_run["dlogz_init"] == 0.01
        assert dynesty.number_of_cores == 4