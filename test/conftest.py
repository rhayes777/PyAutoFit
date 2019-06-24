from os import path

import pytest

import autofit as af

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(autouse=True)
def set_config_path():
    af.conf.instance = af.conf.Config(path.join(directory, "test_files/config"),
                                path.join(directory, "output"))
