import os

import pytest

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