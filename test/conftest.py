from os import path

import pytest

from autofit import conf

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(path.join(directory, "test_files/config"),
                                path.join(directory, "output"))
