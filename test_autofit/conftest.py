from os import path

import pytest

import autofit as af
from autofit import conf
import shutil

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        path.join(directory, "test_files/config"), path.join(directory, "output")
    )


@pytest.fixture(autouse=True)
def remove_output():
    try:
        shutil.rmtree(f"{directory}/output")
    except FileNotFoundError:
        pass


@pytest.fixture
def model():
    return af.ModelMapper()
