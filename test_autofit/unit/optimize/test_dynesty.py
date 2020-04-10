import os

import pytest

from autofit import Paths
from autoconf import conf
import autofit as af

directory = os.path.dirname(os.path.realpath(__file__))
pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        config_path=os.path.join(directory, "files/dynesty/config"),
        output_path=os.path.join(directory, "files/dynesty/output")
    )

class TestDynestyConfig:

    def test__loads_from_config_file_correct(self):

        dynesty = af.Dynesty()

        assert dynesty.terminate_at_acceptance_ratio == True
        assert dynesty.acceptance_ratio_threshold == 2.0

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
        assert copy.terminate_at_acceptance_ratio is optimizer.terminate_at_acceptance_ratio
        assert copy.acceptance_ratio_threshold is optimizer.acceptance_ratio_threshold
